import ctypes
from abc import ABC, abstractmethod, abstractclassmethod
from collections import namedtuple

from .extradims import get_type_for_extra_dim
from .types import Stream
from typing import Iterable, List, Optional, Iterator, Union

NULL_BYTE = b'\x00'


class VLRHeader(ctypes.LittleEndianStructure):
    _fields_ = [
        ('_reserved', ctypes.c_uint16),
        ('user_id', ctypes.c_char * 16),
        ('record_id', ctypes.c_uint16),
        ('record_length_after_header', ctypes.c_uint16),
        ('description', ctypes.c_char * 32)
    ]
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
        data stream
        
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
        header = VLRHeader()
        data_stream.readinto(header)
        raw_vlr.header = header
        raw_vlr.record_data = data_stream.read(header.record_length_after_header)
        return raw_vlr

    def __repr__(self) -> str:
        return 'RawVLR(user_id: {}, record_id: {}, len: {})'.format(
            self.header.user_id, self.header.record_id, self.header.record_length_after_header
        )

class VLR:
    def __init__(self, user_id: str, record_id: int, description: str='', data: bytes=b'') -> None:
        self.user_id = user_id
        self.record_id = record_id
        self.description = description
        self.record_data = bytes(data)

    def into_raw(self) -> RawVLR:
        raw_vlr = RawVLR()
        raw_vlr.header.user_id = self.user_id.encode('utf8')
        raw_vlr.header.description = self.description.encode('utf8')
        raw_vlr.header.record_id = self.record_id
        raw_vlr.header.record_length_after_header = len(self.record_data)
        raw_vlr.record_data = self.record_data

        return raw_vlr

    @classmethod
    def from_raw(cls, raw_vlr: RawVLR) -> 'VLR':
        vlr = cls(
            raw_vlr.header.user_id.rstrip(NULL_BYTE).decode(),
            raw_vlr.header.record_id,
            raw_vlr.header.description.rstrip(NULL_BYTE).decode(),
            raw_vlr.record_data
        )
        return vlr

    def __len__(self) -> int:
        return VLR_HEADER_SIZE + len(self.record_data)

    def __repr__(self) -> str:
        return "{}(user_id: '{}', record_id: '{}', data len: '{}')".format(
            self.__class__.__name__, self.user_id, self.record_id, len(self.record_data))


class KnownVLR(ABC):
    @staticmethod
    @abstractmethod
    def official_user_id() -> str: pass

    @staticmethod
    @abstractmethod
    def official_record_ids() -> Iterable[int]: pass

    @abstractclassmethod
    def from_raw(cls, raw: RawVLR) -> 'KnownVLR': pass


class ClassificationLookupStruct(ctypes.LittleEndianStructure):
    _fields_ = [
        ('class_number', ctypes.c_uint8),
        ('description', ctypes.c_char * 15)
    ]

    def __init__(self, class_number, description):
        if isinstance(description, str):
            super().__init__(class_number, description.encode())
        else:
            super().__init__(class_number, description)

    def __repr__(self):
        return 'ClassificationLookup({} : {})'.format(self.class_number, self.description)

    @staticmethod
    def size():
        return ctypes.sizeof(ClassificationLookupStruct)


class ClassificationLookupVlr(VLR, KnownVLR):
    _lookup_size = ClassificationLookupStruct.size()
    def __init__(self, data=b''):
        super().__init__(self.official_user_id(), self.official_record_ids()[0], description='', data=data)
        self.lookups = []

    def _is_max_num_lookups_reached(self):
        return len(self) >= 256

    def add_lookup(self, class_number, description):
        if not self._is_max_num_lookups_reached():
            self.lookups.append(ClassificationLookupStruct(class_number, description))
        else:
            raise ValueError('Cannot add more lookups')

    def parse_data(self):
        if len(self.record_data) % self._lookup_size != 0:
            raise ValueError("Length of ClassificationLookup VLR's record_data must be a multiple of {}".format(
                self._lookup_size))
        for i in range(len(self.record_data) // ctypes.sizeof(ClassificationLookupStruct)):
            self.lookups.append(ClassificationLookupStruct.from_buffer(
                self.record_data[self._lookup_size * i: self._lookup_size * (i + 1)]))

    def into_raw(self):
        self.record_data = b''.join(bytes(lookup) for lookup in self.lookups)
        return super().into_raw()

    def __len__(self):
        return VLR_HEADER_SIZE + len(self.lookups) * ctypes.sizeof(ClassificationLookupStruct)

    @staticmethod
    def official_user_id():
        return "LASF_Spec"

    @staticmethod
    def official_record_ids():
        return 0,


class LasZipVlr(VLR, KnownVLR):
    def __init__(self, data):
        super().__init__(
            LasZipVlr.official_user_id(),
            LasZipVlr.official_record_ids()[0],
            'http://laszip.org',
            data
        )

    @staticmethod
    def official_user_id():
        return 'laszip encoded'

    @staticmethod
    def official_record_ids():
        return 22204,

    @classmethod
    def from_raw(cls, raw_vlr):
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

    def format_name(self):
        return self.name.rstrip(NULL_BYTE).decode().replace(' ', "_").replace('-', '_')

    def type_tuple(self):
        if self.data_type == 0:
            return self.format_name(), '{}u1'.format(self.options)
        return self.format_name(), get_type_for_extra_dim(self.data_type)

    @staticmethod
    def size():
        return ctypes.sizeof(ExtraBytesStruct)


class ExtraBytesVlr(VLR, KnownVLR):
    def __init__(self, data=b''):
        if (len(data) % ExtraBytesStruct.size()) != 0:
            raise ValueError("Data length of ExtraBytes vlr must be a multiple of {}".format(
                ExtraBytesStruct.size() ))
        super().__init__('LASF_Spec', 4, 'extra_bytes', data)
        self.extra_bytes_structs = []
        self.parse_data()

    def parse_data(self):
        num_extra_bytes_structs = len(self.record_data) // ExtraBytesStruct.size()
        self.extra_bytes_structs = [None] * num_extra_bytes_structs
        for i in range(num_extra_bytes_structs):
            self.extra_bytes_structs[i] = ExtraBytesStruct.from_buffer_copy(self.record_data[ExtraBytesStruct.size() * i: ExtraBytesStruct.size() * (i + 1)])

    def type_of_extra_dims(self):
        return [extra_dim.type_tuple() for extra_dim in self.extra_bytes_structs]

    def __repr__(self):
        return 'ExtraBytesVlr(extra bytes structs: {})'.format(len(self.extra_bytes_structs))

    def into_raw(self):
        self.record_data = b''.join(bytes(extra_struct) for extra_struct in self.extra_bytes_structs)
        return super().into_raw()

    def __len__(self):
        return VLR_HEADER_SIZE + len(self.extra_bytes_structs) * ctypes.sizeof(ExtraBytesStruct)

    @staticmethod
    def official_user_id():
        return 'LASF_Spec'

    @staticmethod
    def official_record_ids():
        return 4,

    @classmethod
    def from_raw(cls, raw_vlr):
        return cls(raw_vlr.record_data)

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
    def size():
        return ctypes.sizeof(WaveformPacketStruct)


class WaveformPacketVlr(VLR, KnownVLR):
    def __init__(self, record_id, data=b''):
        super().__init__(
            self.official_user_id(),
            record_id=record_id,
            description='',
            data=data
        )
        self.parsed_record = None

    @staticmethod
    def official_record_ids():
        return range(100, 356)

    @staticmethod
    def official_user_id():
        return 'LASF_Spec'

    @classmethod
    def from_raw(cls, raw_vlr):
        vlr = cls(
            raw_vlr.header.record_id,
            data=raw_vlr.record_data
        )
        vlr.description = raw_vlr.header.description
        vlr.parsed_record = WaveformPacketStruct.from_buffer_copy(vlr.record_data)
        return vlr



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
