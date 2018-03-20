import ctypes
from abc import ABC, abstractmethod, abstractclassmethod

from .extradims import get_type_for_extra_dim
from .types import Stream
from typing import Iterable, List, Optional, Iterator, Union, Any, Tuple

NULL_BYTE = b'\x00'


class VLRHeader(ctypes.LittleEndianStructure):
    _fields_ = [
        ('_reserved', ctypes.c_uint16),
        ('user_id', ctypes.c_char * 16),
        ('record_id', ctypes.c_uint16),
        ('record_length_after_header', ctypes.c_uint16),
        ('description', ctypes.c_char * 32)
    ]

    @classmethod
    def from_stream(cls, stream):
        return cls.from_buffer(bytearray(stream.read(ctypes.sizeof(cls))))


VLR_HEADER_SIZE = ctypes.sizeof(VLRHeader)


class RawVLR:
    """ As close as possible to the underlying data
    No parsing of the record_data is made
    """

    def __init__(self) -> None:
        self.header = VLRHeader()
        self.record_data = b''

    def write_to(self, out: Stream) -> None:
        """ Write the raw header content to the out stream
        
        Parameters:
        ----------
        out : {file object}
            The output stream
        """

        self.header.record_length_after_header = len(self.record_data)
        out.write(bytes(self.header))
        out.write(self.record_data)

    @classmethod
    def read_from(cls, data_stream: Stream) -> 'RawVLR':
        """ Instanciate a RawVLR by reading the content from the

        Parameters:
        ----------
        data_stream : {file object}
            The input stream
        Returns
        -------
        RawVLR
            The RawVLR read
        """

        raw_vlr = cls()
        header = VLRHeader.from_stream(data_stream)
        raw_vlr.header = header
        raw_vlr.record_data = data_stream.read(header.record_length_after_header)
        return raw_vlr

    def __repr__(self) -> str:
        return 'RawVLR(user_id: {}, record_id: {}, len: {})'.format(
            self.header.user_id, self.header.record_id, self.header.record_length_after_header
        )


class UnknownVLR(ABC):
    @abstractmethod
    def into_raw(self) -> RawVLR: pass

    @abstractmethod
    def __len__(self) -> int: pass

    @abstractclassmethod
    def from_raw(cls, raw: RawVLR) -> 'UnknownVLR': pass


class KnownVLR(UnknownVLR):
    @staticmethod
    @abstractmethod
    def official_user_id() -> str: pass

    @staticmethod
    @abstractmethod
    def official_record_ids() -> Iterable[int]: pass

    @abstractmethod
    def parse_record_data(self, record_data: bytes) -> Any: pass

    @classmethod
    def from_raw(cls, raw: RawVLR) -> 'KnownVLR':
        vlr = cls()
        vlr.parse_record_data(raw.record_data)
        return vlr


class BaseVLR(UnknownVLR):
    def __init__(self, user_id: str, record_id: int, description: Optional[str] = '') -> None:
        self.user_id = user_id
        self.record_id = record_id
        self.description = description

    def into_raw(self) -> RawVLR:
        raw_vlr = RawVLR()
        raw_vlr.header.user_id = self.user_id.encode('utf8')
        raw_vlr.header.description = self.description.encode('utf8')
        raw_vlr.header.record_id = self.record_id
        raw_vlr.record_length_after_header = 0
        raw_vlr.record_data = b''
        return raw_vlr

    def __len__(self) -> int:
        return VLR_HEADER_SIZE


class VLR(BaseVLR):
    def __init__(self, user_id: int, record_id: int, description: Optional[str] = '') -> None:
        super().__init__(user_id, record_id, description=description)
        self.record_data = b''

    def into_raw(self) -> RawVLR:
        raw_vlr = super().into_raw()
        raw_vlr.header.record_length_after_header = len(self.record_data)
        raw_vlr.record_data = self.record_data
        return raw_vlr

    @classmethod
    def from_raw(cls, raw_vlr: RawVLR) -> 'VLR':
        vlr = cls(
            raw_vlr.header.user_id.rstrip(NULL_BYTE).decode(),
            raw_vlr.header.record_id,
            raw_vlr.header.description.rstrip(NULL_BYTE).decode(),
        )
        vlr.record_data = raw_vlr.record_data
        return vlr

    def __len__(self) -> int:
        return VLR_HEADER_SIZE + len(self.record_data)

    def __repr__(self) -> str:
        return "{}(user_id: '{}', record_id: '{}', data len: '{}')".format(
            self.__class__.__name__, self.user_id, self.record_id, len(self.record_data))


class ClassificationLookupStruct(ctypes.LittleEndianStructure):
    _fields_ = [
        ('class_number', ctypes.c_uint8),
        ('description', ctypes.c_char * 15)
    ]

    def __init__(self, class_number: int, description: str) -> None:
        if isinstance(description, str):
            super().__init__(class_number, description.encode())
        else:
            super().__init__(class_number, description)

    def __repr__(self) -> str:
        return 'ClassificationLookup({} : {})'.format(self.class_number, self.description)

    @staticmethod
    def size() -> int:
        return ctypes.sizeof(ClassificationLookupStruct)


class ClassificationLookupVlr(BaseVLR, KnownVLR):
    _lookup_size = ClassificationLookupStruct.size()

    def __init__(self) -> None:
        super().__init__(self.official_user_id(), self.official_record_ids()[0], description='')
        self.lookups = []

    def _is_max_num_lookups_reached(self) -> bool:
        return len(self) >= 256

    def add_lookup(self, class_number: int, description: str) -> None:
        if not self._is_max_num_lookups_reached():
            self.lookups.append(ClassificationLookupStruct(class_number, description))
        else:
            raise ValueError('Cannot add more lookups')

    def into_raw(self) -> RawVLR:
        raw = super().into_raw()
        raw.record_data = b''.join(bytes(lookup) for lookup in self.lookups)
        return raw

    def __len__(self) -> int:
        return VLR_HEADER_SIZE + len(self.lookups) * ctypes.sizeof(ClassificationLookupStruct)

    def parse_record_data(self, record_data: bytes) -> None:
        if len(record_data) % self._lookup_size != 0:
            raise ValueError("Length of ClassificationLookup VLR's record_data must be a multiple of {}".format(
                self._lookup_size))
        for i in range(len(record_data) // ctypes.sizeof(ClassificationLookupStruct)):
            self.lookups.append(ClassificationLookupStruct.from_buffer(
                record_data[self._lookup_size * i: self._lookup_size * (i + 1)]))

    @staticmethod
    def official_user_id() -> str:
        return "LASF_Spec"

    @staticmethod
    def official_record_ids() -> Tuple[int]:
        return 0,


class LasZipVlr(VLR, KnownVLR):
    def __init__(self, data: bytes):
        super().__init__(
            LasZipVlr.official_user_id(),
            LasZipVlr.official_record_ids()[0],
            'http://laszip.org',
        )
        self.record_data = data

    def parse_record_data(self, record_data):
        # Only laz-perf/laszip knows how to parse this
        pass

    @staticmethod
    def official_user_id() -> str:
        return 'laszip encoded'

    @staticmethod
    def official_record_ids() -> Tuple[int]:
        return 22204,

    @classmethod
    def from_raw(cls, raw_vlr: RawVLR) -> 'LasZipVlr':
        return cls(raw_vlr.record_data)


class ExtraBytesStruct(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('reserved', ctypes.c_uint8 * 2),
        ('data_type', ctypes.c_uint8),
        ('options', ctypes.c_uint8),
        ('name', ctypes.c_char * 32),
        ('unused', ctypes.c_uint8 * 4),
        ('no_data', ctypes.c_double * 3),
        ('min', ctypes.c_double * 3),
        ('max', ctypes.c_double * 3),
        ('scale', ctypes.c_double * 3),
        ('offset', ctypes.c_double * 3),
        ('description', ctypes.c_char * 32),
    ]

    def format_name(self) -> str:
        return self.name.rstrip(NULL_BYTE).decode().replace(' ', "_").replace('-', '_')

    def type_tuple(self) -> Tuple[str, str]:
        if self.data_type == 0:
            return self.format_name(), '{}u1'.format(self.options)
        return self.format_name(), get_type_for_extra_dim(self.data_type)

    @staticmethod
    def size() -> int:
        return ctypes.sizeof(ExtraBytesStruct)


class ExtraBytesVlr(BaseVLR, KnownVLR):
    def __init__(self) -> None:
        super().__init__('LASF_Spec', self.official_record_ids()[0], 'extra_bytes')
        self.extra_bytes_structs = []

    def parse_record_data(self, data: bytes) -> None:
        if (len(data) % ExtraBytesStruct.size()) != 0:
            raise ValueError("Data length of ExtraBytes vlr must be a multiple of {}".format(
                ExtraBytesStruct.size()))
        num_extra_bytes_structs = len(data) // ExtraBytesStruct.size()
        self.extra_bytes_structs = [None] * num_extra_bytes_structs
        for i in range(num_extra_bytes_structs):
            self.extra_bytes_structs[i] = ExtraBytesStruct.from_buffer_copy(
                data[ExtraBytesStruct.size() * i: ExtraBytesStruct.size() * (i + 1)])

    def type_of_extra_dims(self) -> Tuple[str, str]:
        return [extra_dim.type_tuple() for extra_dim in self.extra_bytes_structs]

    def __repr__(self) -> str:
        return 'ExtraBytesVlr(extra bytes structs: {})'.format(len(self.extra_bytes_structs))

    def into_raw(self) -> RawVLR:
        raw = super().into_raw()
        raw.record_data = b''.join(bytes(extra_struct) for extra_struct in self.extra_bytes_structs)
        return raw

    def __len__(self) -> int:
        return VLR_HEADER_SIZE + len(self.extra_bytes_structs) * ExtraBytesStruct.size()

    @staticmethod
    def official_user_id() -> str:
        return 'LASF_Spec'

    @staticmethod
    def official_record_ids() -> Tuple[int]:
        return 4,


class WaveformPacketStruct(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('bits_per_sample', ctypes.c_ubyte),
        ('waveform_compression_type', ctypes.c_ubyte),
        ('number_of_samples', ctypes.c_uint32),
        ('temporal_sample_spacing', ctypes.c_uint32),
        ('digitizer_gain', ctypes.c_double),
        ('digitizer_offset', ctypes.c_double)
    ]

    @staticmethod
    def size() -> int:
        return ctypes.sizeof(WaveformPacketStruct)


class WaveformPacketVlr(BaseVLR, KnownVLR):
    def __init__(self, record_id, description=''):
        super().__init__(
            self.official_user_id(),
            record_id=record_id,
            description=description,
        )
        self.parsed_record = None

    def into_raw(self) -> RawVLR:
        raw = super().into_raw()
        raw.record_data = bytes(self.parsed_record)
        return raw

    def __len__(self) -> int:
        return super().__len__() + WaveformPacketStruct.size()

    @staticmethod
    def official_record_ids() -> Iterable[int]:
        return range(100, 356)

    @staticmethod
    def official_user_id() -> str:
        return 'LASF_Spec'

    @classmethod
    def from_raw(cls, raw_vlr: RawVLR) -> 'WaveformPacketVlr':
        vlr = cls(raw_vlr.header.record_id, description=raw_vlr.header.description.decode())
        vlr.description = raw_vlr.header.description
        vlr.parsed_record = WaveformPacketStruct.from_buffer_copy(raw_vlr.record_data)
        return vlr


class GeoKeyEntryStruct(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('id', ctypes.c_uint16),
        ('tiff_tag_location', ctypes.c_uint16),
        ('count', ctypes.c_uint16),
        ('value_offset', ctypes.c_uint16),
    ]

    @staticmethod
    def size():
        return ctypes.sizeof(GeoKeysHeaderStructs)

    def __repr__(self):
        return 'GeoKey(Id: {}, Location: {}, count: {}, offset: {})'.format(
            self.id, self.tiff_tag_location, self.count, self.value_offset
        )


class GeoKeysHeaderStructs(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ('key_direction_version', ctypes.c_uint16),
        ('key_revision', ctypes.c_uint16),
        ('minor_revision', ctypes.c_uint16),
        ('number_of_keys', ctypes.c_uint16),
    ]

    def __init__(self):
        super().__init__(
            key_directory_version=1,
            key_revision=1,
            minor_revision=0,
            number_of_kets=0
        )

    @staticmethod
    def size():
        return ctypes.sizeof(GeoKeysHeaderStructs)

    def __repr__(self):
        return 'GeoKeysHeader(vers: {}, rev:{}, minor: {}, num_keys: {})'.format(
            self.key_direction_version, self.key_revision, self.minor_revision,
            self.number_of_keys
        )


class GeoKeyDirectoryVlr(BaseVLR, KnownVLR):
    def __init__(self):
        super().__init__(
            self.official_user_id(),
            self.official_record_ids()[0],
            description=''
        )
        self.geo_keys_header = GeoKeysHeaderStructs()
        self.geo_keys = [GeoKeyEntryStruct()]

    def parse_record_data(self, record_data):
        record_data = bytearray(record_data)
        header_data = record_data[:ctypes.sizeof(GeoKeysHeaderStructs)]
        self.geo_keys_header = GeoKeysHeaderStructs.from_buffer(header_data)
        self.geo_keys, keys_data = [], record_data[ctypes.sizeof(GeoKeysHeaderStructs):]

        for i in range(self.geo_keys_header.number_of_keys):
            data = keys_data[(i * GeoKeyEntryStruct.size()): (i + 1) * GeoKeyEntryStruct.size()]
            self.geo_keys.append(GeoKeyEntryStruct.from_buffer(data))

    def into_raw(self):
        raw = super().into_raw()
        raw.record_data = bytes(self.geo_keys_header)
        raw.record_data += b''.join(map(bytes, self.geo_keys))
        return raw

    def __len__(self):
        return VLR_HEADER_SIZE + GeoKeysHeaderStructs.size() + len(self.geo_keys) * GeoKeyEntryStruct.size()

    @staticmethod
    def official_user_id():
        return 'LASF_Projection'

    @staticmethod
    def official_record_ids():
        return 34735,


class GeoDoubleParamsVlr(BaseVLR, KnownVLR):
    def __init__(self):
        super().__init__(
            self.official_user_id(),
            self.official_record_ids()[0],
            description=''
        )
        self.doubles = []

    def parse_record_data(self, record_data):
        sizeof_double = ctypes.sizeof(ctypes.c_double)
        if len(record_data) % sizeof_double != 0:
            raise ValueError("GeoDoubleParams record data length () is not a multiple of sizeof(double) ()".format(
                len(record_data), sizeof_double
            ))
        record_data = bytearray(record_data)
        num_doubles = len(record_data) // sizeof_double
        for i in range(num_doubles):
            b = record_data[i * sizeof_double:(i + 1) * sizeof_double]
            self.doubles.append(ctypes.c_double.from_buffer(b))

    def __len__(self):
        return VLR_HEADER_SIZE + len(self.doubles) * ctypes.sizeof(ctypes.c_double)

    def into_raw(self):
        raw = super().into_raw()
        raw.record_data = b''.join(map(bytes, self.doubles))
        return raw

    @staticmethod
    def official_user_id():
        return 'LASF_Projection'

    @staticmethod
    def official_record_ids():
        return 34736,


class GeoAsciiParamsVlr(BaseVLR, KnownVLR):
    def __init__(self):
        super().__init__(
            self.official_user_id(),
            self.official_record_ids()[0],
            description=''
        )
        self.strings = []

    def parse_record_data(self, record_data):
        self.strings = [s.decode('ascii') for s in record_data.split(NULL_BYTE)]

    def into_raw(self):
        raw = super().into_raw()
        raw.record_data = NULL_BYTE.join(s.encode('ascii') for s in self.strings)
        return raw

    def __len__(self):
        return VLR_HEADER_SIZE + sum(map(len, self.strings)) + len(NULL_BYTE) * (len(self.strings) - 1)

    @staticmethod
    def official_user_id():
        return 'LASF_Projection'

    @staticmethod
    def official_record_ids():
        return 34737,


def vlr_factory(raw_vlr: RawVLR) -> Union[VLR, KnownVLR]:
    user_id = raw_vlr.header.user_id.rstrip(NULL_BYTE).decode()
    for known_vlr in KnownVLR.__subclasses__():
        if known_vlr.official_user_id() == user_id and raw_vlr.header.record_id in known_vlr.official_record_ids():
            return known_vlr.from_raw(raw_vlr)
    else:
        return VLR.from_raw(raw_vlr)


class VLRList:
    def __init__(self) -> None:
        self.vlrs = []

    def append(self, vlr: VLR) -> None:
        self.vlrs.append(vlr)

    def get_by_id(self, user_id='', record_ids=(None,)):
        if user_id != '' and record_ids != (None,):
            return [vlr for vlr in self.vlrs if vlr.user_id == user_id and vlr.record_id in record_ids]
        else:
            return [vlr for vlr in self.vlrs if vlr.user_id == user_id or vlr.record_id in record_ids]

    def get(self, vlr_type: str) -> List[VLR]:
        return [v for v in self.vlrs if v.__class__.__name__ == vlr_type]

    def extract(self, vlr_type: str) -> List[VLR]:
        kept_vlrs, extracted_vlrs = [], []
        for vlr in self.vlrs:
            if vlr.__class__.__name__ == vlr_type:
                extracted_vlrs.append(vlr)
            else:
                kept_vlrs.append(vlr)
        self.vlrs = kept_vlrs
        return extracted_vlrs

    def pop(self, index: int) -> VLR:
        return self.vlrs.pop(index)

    def index(self, vlr_type: str) -> int:
        for i, v in enumerate(self.vlrs):
            if v.__class__.__name__ == vlr_type:
                return i
        else:
            raise ValueError('{} is not in the VLR list'.format(vlr_type))

    def write_to(self, out: Stream) -> None:
        for vlr in self.vlrs:
            vlr.into_raw().write_to(out)

    def total_size_in_bytes(self) -> int:
        return sum(len(vlr) for vlr in self.vlrs)

    def __iter__(self) -> Iterator[VLR]:
        yield from iter(self.vlrs)

    def __getitem__(self, item) -> VLR:
        return self.vlrs[item]

    def __len__(self) -> int:
        return len(self.vlrs)

    def __eq__(self, other: Union[List[VLR], 'VLRList']) -> bool:
        if isinstance(other, list):
            return self.vlrs == other
        else:
            return self.vlrs == other.vlrs

    def __repr__(self):
        return "[{}]".format(", ".join(repr(vlr) for vlr in self.vlrs))

    @classmethod
    def read_from(cls, data_stream: Stream, num_to_read: int) -> 'VLRList':
        vlrlist = cls()
        for _ in range(num_to_read):
            raw = RawVLR.read_from(data_stream)
            try:
                vlrlist.append(vlr_factory(raw))
            except UnicodeDecodeError:
                print("Failed to decode VLR: {}".format(raw))

        return vlrlist

    @classmethod
    def from_list(cls, vlr_list: List[VLR]) -> 'VLRList':
        vlrs = cls()
        vlrs.vlrs = vlr_list
        return vlrs
