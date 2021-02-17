import os
import logging
import zstandard as zstd

MAX_LEVEL = 20
contexts = [zstd.ZstdCompressor(level=level) for level in range(1, MAX_LEVEL + 1)]
decompressor_context = zstd.ZstdDecompressor()


def compressed_filename(base_filename):
    return base_filename + '.zst'

def is_compressed(filename):
    return os.path.splitext(filename)[-1] == '.zst'

def compress(gop_id, level):
    from vfs.gop import Gop
    from vfs.engine import VFS

    assert(level > 0)

    context = contexts[level - 1]
    gop = Gop.get(gop_id)
    zstd_filename = compressed_filename(gop.filename)

    if not gop.zstandard is None:
        decompress(gop)

    assert(gop.zstandard is None)

    logging.info(f'Compressing GOP {gop.id} at level {level}')

    with open(gop.filename, 'rb') as input, open(zstd_filename, 'wb') as out:
        bytes_read, bytes_written = context.copy_stream(input, out)

    VFS.instance().database.execute('UPDATE gops SET zstandard = ?, size = ? WHERE id = ?',
                                    (level, bytes_written, gop_id)).close()
    os.remove(gop.filename)

def decompress(gop):
    from vfs.engine import VFS

    assert(gop.zstandard is not None)

    zstd_filename = compressed_filename(gop.filename)

    logging.info(f'Decompressing GOP {gop.id}')

    with open(zstd_filename, 'rb') as input, open(gop.filename, 'wb') as out:
        bytes_read, bytes_written = decompressor_context.copy_stream(input, out)

    VFS.instance().database.execute('UPDATE gops SET zstandard = NULL, size = ? WHERE id = ?',
                                    (bytes_written, gop.id)).close()
    gop.zstandard = None
    gop.size = bytes_written
    os.remove(zstd_filename)

def decompress_file(filename, output_stream):
    logging.info(f'Decompressing {filename}')

    with open(filename, 'rb') as input:
        return decompressor_context.copy_stream(input, output_stream)
