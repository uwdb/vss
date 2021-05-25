import os
from multiprocessing.pool import Pool
from os import path
from itertools import groupby
from concurrent.futures import ProcessPoolExecutor, wait
import tempfile
import logging

from videoio import encoded

from vfs.engine import VFS
from vfs.physicalvideo import PhysicalVideo
from vfs.rawcompression import compressed_filename
from vfs.gop import Gop
import vfs.videoio
import vfs.rawcompression
from vfs.mp4 import MP4
from vfs.utilities import log_runtime

POOL_SIZE = 24
pool = None

class ReconstructionSegment(object):
    def __init__(self, gops, resolution, codec, times):
        self.gops = list(gops)
        self.transcode_filename = None
        self.transcode_size = None
        self.new_mse = None
        self.requires_transcode = self._requires_transcode(self.gops[0], resolution, codec, times)

    @staticmethod
    def _requires_transcode(gop, resolution, codec, times):
        gop_times = (max(times[0] - gop.start_time, 0),
                     (min(gop.end_time, times[1]) - gop.start_time)) if times else None

        return gop.video().resolution() != resolution or \
               gop.video().codec != codec or \
               (times is None or 0 < gop_times[0] < gop_times[1] < gop.end_time - gop.start_time)

    @property
    def video(self):
        return self[0].video()

    @property
    def filename(self):
        return self.transcode_filename if self.requires_transcode else self.video.filename

    def __getitem__(self, index):
        return self.gops[index]

class OptionalTemporaryDirectory(object):
    def __init__(self, requires_transcode):
        self.directory = tempfile.TemporaryDirectory() if requires_transcode else None

    def __enter__(self):
        if self.directory is not None:
            return self.directory.__enter__()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if self.directory is not None:
            self.directory.__exit__(exception_type, exception_value, exception_traceback)


def reconstruct_gops(sequence, temp_path, times, resolution, codec, roi, fps): #, filenames, cache_sequences):
    with log_runtime(f'Reconstruct GOP segment {sequence.video.id}.{sequence[0].id}-{sequence.video.id}.{sequence[-1].id} ({sequence.video.width}x{sequence.video.height}, {sequence.video.codec}, t={sequence[0].start_time:0.2f}-{sequence[-1].end_time:0.2f})'):
        if sequence[0].joint:
            assert(all(gop.joint for gop in sequence))
            # Code to correct codec/resolution to skip transcode
            # TODO BH this is broken after group update, need to subselect byte range before codecompressing
            VFS.instance().compression.co_decompress(sequence, sequence.video.filename)
        if sequence[0].zstandard:
            assert(all(gop.zstandard for gop in sequence))
            logging.debug(f"Reconstruction: decompressing raw GOP {','.join(str(gop.id) for gop in gops)}")
            # TODO BH this is broken after group update
            vfs.rawcompression.decompress(sequence[0])

        video_times = (max(times[0] - sequence.video.start_time(), 0),
                     (min(sequence.video.end_time(), times[1]) - sequence.video.start_time())) if times else None

        container = '.mp4' if encoded[codec] else ''

        sequence.transcode_filename = path.join(temp_path, 'resize-{}.{}{}'.format(path.basename(sequence.video.filename), codec, container))
        sequence.new_mse = vfs.videoio.reformat(
            sequence.video.filename, sequence.transcode_filename,
            input_resolution=sequence.video.resolution(), output_resolution=resolution,
            input_codec=sequence.video.codec, output_codec=codec,
            input_fps=sequence[0].fps, output_fps=fps,
            roi=roi,
            times=video_times if video_times != (sequence.video.start_time, sequence.video.end_time) else None)
        sequence.transcode_size = os.path.getsize(sequence.transcode_filename)

        return sequence

def reconstruct(output_filename, logical, gops, resolution, roi, times, fps, codec, is_stream=False):
    global pool # Move to VFS.engine
    fps = fps or gops[0].fps
    segments = [ReconstructionSegment(g, resolution, codec, times) for k, g in
                groupby(gops, key=lambda gop: (gop.physical_id, gop.id if gop.joint or gop.zstandard else None))]
    has_transcode = any(s.requires_transcode for s in segments)

    assert(len(segments) > 0)

    if pool is None:
        pool = Pool(POOL_SIZE)

    with OptionalTemporaryDirectory(has_transcode) as temp_path:
        transcode_segments = pool.starmap(reconstruct_gops,
                                          ((segment, temp_path, times, resolution, codec, roi, fps)
                                          for segment in segments if segment.requires_transcode))

        segments = [segment if not segment.requires_transcode else transcode_segments.pop(0) for segment in segments]

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

        if is_stream and len(segments) == 1 and (times is None or times == (segments[0].video.start_time(), segments[0].video.end_time())):
            output_filename = segments[0].video.filename
        else:
            vfs.videoio.join_video(segments, output_filename, resolution, codec) #, input_sizes=sizes)

            cache_reconstructions(logical, resolution, codec, times, fps,
                                  (segment for segment in segments if segment.requires_transcode))

    return output_filename

def cache_reconstructions(logical, resolution, codec, times, fps, cache_sequences):
    for sequence in cache_sequences:
        physical = PhysicalVideo.load(logical, sequence.transcode_filename, resolution=resolution, codec=codec, fps=fps, copy=False)

        transitive_estimated_mses = VFS.instance().database.execute(
            'WITH RECURSIVE'
            '  error(id, estimated_mse) AS '
            '    (SELECT id, estimated_mse FROM gops child WHERE id IN ({}) '.format(
                ','.join(str(gop.id) for gop in sequence.gops)) +
            '    UNION ALL '
            '    SELECT id, estimated_mse FROM gops parent WHERE parent.id = id)'
            'SELECT id, SUM(estimated_mse) FROM error GROUP BY id').fetchall()

        batch = [f'UPDATE gops SET estimated_mse = {2 * (current_estimated_mse + sequence.new_mse or 0)} WHERE id = {id}'
                 for (id, current_estimated_mse) in transitive_estimated_mses]
        VFS.instance().database.executebatch(batch)

        logging.info('Cached physical video %s-%d', logical.name, physical.id)