import os
import zstandard as zstd
from vfs.engine import VFS
from vfs.gop import Gop

MAX_LEVEL = 20
contexts = [zstd.ZstdCompressor(level=level) for level in range(1, MAX_LEVEL + 1)]
decompressor_context = zstd.ZstdDecompressor()


def _zstd_filename(base_filename):
    return base_filename + '.zstd'

def compress(gop_id, level):
    context = contexts[level - 1]
    gop = Gop.get(gop_id)
    zstd_filename = _zstd_filename(gop.filename)


    with open(gop.filename, 'rb') as input, open(zstd_filename, 'wb') as out:
        bytes_read, bytes_written = context.copy_stream(input, out)

    VFS.instance().database.execute('UPDATE gops SET zstandard = ?, size = ? WHERE id = ?',
                                    (level, bytes_written, gop_id)).close()
    os.remove(gop.filename)

def decompress(gop):
    zstd_filename = _zstd_filename(gop.filename)

    with open(zstd_filename, 'rb') as input, open(gop.filename, 'wb') as out:
        bytes_read, bytes_written = decompressor_context.copy_stream(input, out)

    VFS.instance().database.execute('UPDATE gops SET zstandard = NULL, size = ? WHERE id = ?',
                                    (bytes_written, gop.id)).close()
    gop.zstandard = None
    gop.size = bytes_written
    os.remove(zstd_filename)
