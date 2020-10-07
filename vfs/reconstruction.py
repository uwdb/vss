import os
from os import path
import tempfile
import logging
from vfs.engine import VFS
from vfs.physicalvideo import PhysicalVideo
from vfs.gop import Gop
import vfs.videoio
import vfs.rawcompression

def reconstruct(output_filename, logical, gops, resolution, roi, times, fps, codec):
    filenames = []
    cache_sequences = [[]]
    fps = fps or gops[0].fps

    with tempfile.TemporaryDirectory() as temp_path:
        for gop in gops:
            if not gop.joint:
                input_filename = gop.filename
            else:
                input_filename = path.join(temp_path, path.basename(gop.filename.format('original')))
                VFS.instance().compression.co_decompress(gop, input_filename)

            if gop.zstandard:
                logging.debug("Reconstruction: decompressing raw GOP %d", gop.id)
                vfs.rawcompression.decompress(gop)

            gop_times = (max(times[0] - gop.start_time, 0),
                         (min(gop.end_time, times[1]) - gop.start_time)) if times else None

            if (gop.video().resolution != resolution or
                    gop().video().codec != codec or
                    (roi is not None and roi != (0, 0, *resolution)) or
                    (times is None or 0 < gop_times[0] < gop_times[1] < gop.end_time - gop.start_time)):
                    resize_filename = path.join(temp_path, 'resize-{}.{}'.format(path.basename(input_filename), codec))
                    new_mse = vfs.videoio.reformat(input_filename, resize_filename,
                                     input_resolution=gop.video().resolution(), output_resolution=resolution,
                                     input_codec=gop.video().codec, output_codec=codec,
                                     input_fps=gop.fps, output_fps=fps,
                                     roi=roi,
                                     times=gop_times)
                    filenames.append(resize_filename)
                    cache_sequences[-1].append((gop, resize_filename, new_mse))
            else:
                filenames.append(input_filename)
                cache_sequences.append([])

        vfs.videoio.join_video(filenames, output_filename, resolution, codec)

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

        if cache_sequences != [[]]:
            cache_reconstructions(logical, resolution, codec, times, fps, cache_sequences)

        return output_filename

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
            os.rename(filename, new_filename)

        Gop.addmany(physical, new_gop_data)
        logging.info('Cached physical video %s-%d', logical.name, physical.id)
