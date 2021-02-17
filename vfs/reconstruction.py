import os
import uuid
from multiprocessing.pool import Pool
from os import path
from concurrent.futures import ProcessPoolExecutor, wait
import tempfile
import logging
from shutil import copyfile

from videoio import encoded

from vfs.engine import VFS
from vfs.physicalvideo import PhysicalVideo
from vfs.rawcompression import compressed_filename
from vfs.gop import Gop
import vfs.videoio
import vfs.rawcompression
from vfs.utilities import log_runtime

POOL_SIZE = 24
pool = None

def reconstruct_gop(gop, temp_path, times, resolution, codec, roi, fps): #, filenames, cache_sequences):
    with log_runtime(f'Reconstruct GOP {gop.video().id}.{gop.id} ({gop.video().width}x{gop.video().height}, {gop.video().codec}, t={gop.start_time:0.2f}-{gop.end_time:0.2f})'):
        if not gop.joint:
            input_filename = gop.filename
        else:
            input_filename = path.join(temp_path, path.basename(gop.filename.format('original')))
            # Code to correct codec/resolution to skip transcode
            VFS.instance().compression.co_decompress(gop, input_filename)

        gop_times = (max(times[0] - gop.start_time, 0),
                     (min(gop.end_time, times[1]) - gop.start_time)) if times else None

        if (gop.video().resolution() != resolution or
                gop.video().codec != codec or
                #TODO (roi is not None and roi != (0, 0, *resolution)) or
                (times is None or 0 < gop_times[0] < gop_times[1] < gop.end_time - gop.start_time)):
            if gop.zstandard:
                logging.debug("Reconstruction: decompressing raw GOP %d", gop.id)
                vfs.rawcompression.decompress(gop)

            container = '.mp4' if encoded[codec] else ''
            resize_filename = path.join(temp_path, 'resize-{}.{}{}'.format(path.basename(input_filename), codec, container))
            new_mse = vfs.videoio.reformat(input_filename, resize_filename,
                                           input_resolution=gop.video().resolution(), output_resolution=resolution,
                                           input_codec=gop.video().codec, output_codec=codec,
                                           input_fps=gop.fps, output_fps=fps,
                                           roi=roi,
                                           times=gop_times if gop_times != (gop.start_time, gop.end_time) else None)
            return resize_filename, os.path.getsize(resize_filename), (gop, resize_filename, new_mse)
        else:
            logging.info(f"Cache hit for GOP {gop.id}")
            return input_filename if gop.zstandard is None else compressed_filename(input_filename), gop.original_size, []


def reconstruct(output_filename, logical, gops, resolution, roi, times, fps, codec, is_stream=False):
    global pool # Move to VFS.engine
    fps = fps or gops[0].fps
    futures = []

    #with tempfile.TemporaryDirectory() as temp_path:
    #with ProcessPoolExecutor(max_workers=POOL_SIZE) as pool, tempfile.TemporaryDirectory() as temp_path:
    if pool is None:
        pool = Pool(POOL_SIZE)

    with tempfile.TemporaryDirectory() as temp_path:
    #with Pool(POOL_SIZE) as pool, tempfile.TemporaryDirectory() as temp_path:
        filenames, sizes, cache_sequences = zip(*pool.starmap(reconstruct_gop, ((gop, temp_path, times, resolution, codec, roi, fps) for gop in gops)))
        #for gop in gops:
        #    futures.append(pool.submit(reconstruct_gop, gop, temp_path, times, resolution, codec, roi, fps)) #, filenames, cache_sequences))
        #wait(futures)
        #filenames, sizes, cache_sequences = zip(*(f.result() for f in futures))

        accumulated_sequences = [[]]
        for sequence in cache_sequences:
            if not sequence:
                accumulated_sequences.append([])
            else:
                accumulated_sequences[-1].append(sequence)
        if accumulated_sequences != [[]] and accumulated_sequences[0] == []:
            accumulated_sequences.pop(0)

        if len(filenames) == 0:
            assert False
        elif len(filenames) > 1:
            vfs.videoio.join_video(filenames, output_filename, resolution, codec, input_sizes=sizes)
        #elif is_stream and os.path.split(filenames[0])[0] == VFS.instance().path:
            #os.symlink(path.join(os.getcwd(), filenames[0]), output_filename)
        #    output_filename = filenames[0]

        VFS.instance().clock += 1
        clock = VFS.instance().clock

        logging.info("Wrote %s to %s (%dx%d %s%s, times=%s, fps=%d, clock=%d)",
                     logical.name, output_filename,
                     resolution[1], resolution[0],
                     codec,
                     ', roi={}'.format(roi) if roi else '',
                     times,
                     fps,
                     clock)

        VFS.instance().database.execute('UPDATE gops SET clock=? WHERE id IN ({})'.format(','.join(str(g.id) for g in gops)), clock)

        if roi is None and accumulated_sequences != [[]]:
            new_gops = cache_reconstructions(logical, resolution, codec, times, fps, accumulated_sequences)
        else:
            new_gops = []

        if is_stream and len(new_gops) == 1:
            output_filename = new_gops[0][0]
        elif len(filenames) == 1 and is_stream and os.path.split(filenames[0])[0] == VFS.instance().path:
            output_filename = filenames[0]
        elif len(filenames) == 1:
            copyfile(filenames[0], output_filename)

    return output_filename

#def stream_reconstruct(logical, gops, resolution, roi, times, fps, codec):
#    fifo_name = path.join(tempfile.gettempdir(), uuid.uuid4().hex)
#    os.mkfifo(fifo_name)

    #pool.submit(reconstruct, fifo_name, logical, gops, resolution, roi, times, fps, codec)
#    reconstruct(fifo_name, logical, gops, resolution, roi, times, fps, codec)
#    return open(fifo_name, "r")

def cache_reconstructions(logical, resolution, codec, times, fps, cache_sequences):
    for sequence in (sequence for sequence in cache_sequences if sequence):
        physical = PhysicalVideo.add(logical, *resolution, codec)
        new_gop_data = []

        transitive_estimated_mses = VFS.instance().database.execute(
            'WITH RECURSIVE'
            '  error(id, estimated_mse) AS '
            '    (SELECT id, estimated_mse FROM gops child WHERE id IN ({}) '.format(
                ','.join(str(gop.id) for gop, _, _ in sequence)) +
            '    UNION ALL '
            '    SELECT id, estimated_mse FROM gops parent WHERE parent.id = id)'
            'SELECT id, SUM(estimated_mse) FROM error GROUP BY id').fetchall()

        for (_, current_estimated_mse), (index, (gop, filename, new_mse)) in zip(transitive_estimated_mses, enumerate(sequence)):
            new_filename = path.join(VFS.instance().path,
                                          PhysicalVideo._gop_filename_template(logical, physical, index))

            new_gop_data.append((new_filename,
                                 max(gop.start_time, times[0]),
                                 min(gop.end_time, times[1]),
                                 os.path.getsize(filename),
                                 fps,
                                 None,
                                 2 * (current_estimated_mse + (new_mse or 0)),
                                 gop.id))

            #TODO remove guard
            #duplicates = VFS.instance().database.execute("SELECT * FROM gops, physical_videos WHERE start_time = ? AND end_time = ? and height = ? and width = ? and codec = ? and gops.physical_id = physical_videos.id",
            #                                (new_gop_data[-1][0], new_gop_data[-1][1], resolution[0], resolution[1], codec)).fetchall()
            #if len(duplicates) > 0:
            #    print("Duplicate detected???")

            os.rename(filename, new_filename)

        Gop.addmany(physical, new_gop_data)
        logging.info('Cached physical video %s-%d', logical.name, physical.id)
        return new_gop_data
