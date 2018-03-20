import numpy as np

from .errors import LazPerfNotFound
from .point.dims import get_dtype_of_format_id, POINT_FORMAT_DIMENSIONS
from . import vlr

HAS_LAZPERF = False

try:
    import lazperf

    HAS_LAZPERF = True
except ModuleNotFoundError:
    HAS_LAZPERF = False


def raise_if_no_lazperf() -> None:
    if not HAS_LAZPERF:
        raise LazPerfNotFound('Lazperf not found, cannot manipulate laz data')


def is_point_format_compressed(point_format_id: int) -> bool:
    try:
        compression_bit_7 = (point_format_id & 0x80) >> 7
        compression_bit_6 = (point_format_id & 0x40) >> 6
        if not compression_bit_6 and compression_bit_7:
            return True
    except ValueError:
        pass
    return False


def compressed_id_to_uncompressed(point_format_id: int) -> int:
    return point_format_id & 0x3f


def uncompressed_id_to_compressed(point_format_id: int) -> int:
    return (2 ** 7) | point_format_id


def decompress_buffer(
        compressed_buffer: bytes,
        point_format_id: int,
        point_count: int,
        laszip_vlr: vlr.LasZipVlr
    ) -> np.ndarray:
    raise_if_no_lazperf()

    ndtype = get_dtype_of_format_id(point_format_id)
    point_compressed = np.frombuffer(compressed_buffer, dtype=np.uint8)

    vlr_data = np.frombuffer(laszip_vlr.record_data, dtype=np.uint8)
    decompressor = lazperf.VLRDecompressor(point_compressed, vlr_data)

    point_uncompressed = decompressor.decompress_points(point_count)

    point_uncompressed = np.frombuffer(point_uncompressed, dtype=ndtype, count=point_count)

    return point_uncompressed


def create_laz_vlr(point_format_id: int) -> lazperf.LazVLR:
    raise_if_no_lazperf()
    record_schema = lazperf.RecordSchema()

    if 'gps_time' in POINT_FORMAT_DIMENSIONS[point_format_id]:
        record_schema.add_gps_time()

    if 'red' in POINT_FORMAT_DIMENSIONS[point_format_id]:
        record_schema.add_rgb()

    return lazperf.LazVLR(record_schema)


def compress_buffer(uncompressed_buffer: bytes, record_schema: lazperf.RecordSchema, offset: int) -> np.ndarray:
    raise_if_no_lazperf()

    compressor = lazperf.VLRCompressor(record_schema, offset)
    uncompressed_buffer = np.frombuffer(uncompressed_buffer, dtype=np.uint8)
    compressed = compressor.compress(uncompressed_buffer)

    return compressed
