import numpy as np
import logging

from vfs.engine import VFS
from vfs.videoio import HEVC, H264, RGB8
from vfs.gop import Gop
from vfs.constraints import build_from_video_info, find_best_intervals
from vfs.utilities import log_runtime

alpha = {
    (H264, HEVC): 1,
    (H264, RGB8): 0.288,
    (HEVC, H264): 0.871,
    (HEVC, RGB8) : 0.315,
    (RGB8, HEVC): 0.501,
    (RGB8, H264): 0.488
}
eta = 1.45

def _between(gop, t):
    start, end = t if t else (None, None)
    return (t is None or
            gop.start_time <= start < gop.end_time or
            (start <= gop.start_time and gop.end_time <= end) or
            gop.start_time < end <= gop.end_time)

#c_t
def _transcode_cost(resolution, source_codec, target_codec):
    return np.prod(resolution) * alpha[(source_codec, target_codec)]

#c_l
def _lookback_cost(start_frame, frame_count):
    return start_frame == 1 + eta * frame_count

def _prepare(logical, resolution, t):
    """
    physical_videos = {physical.id:
        {'id': physical.id,
         'start': physical.start_time(),
         'end': physical.end_time(),
         'format': physical.codec,
         'clock': 0,
         'size': physical.size(),
         'time': None,
         'resolution': physical.resolution(),
         'frames': sum(gop.fps for gop in physical.gops()) }
        for physical in logical.videos()
        if t is None or
               (physical.start_time() is not None and physical.end_time() is not None and
                physical.start_time() < t[1] and physical.end_time() >= t[0]) and
               #(physical.start_time() or 0) <= t[0] and
               #(physical.end_time() or t[1]) >= t[1]) and
           #physical.start_time() is not None and
           physical.resolution()[0] >= resolution[0] and
           physical.resolution()[1] >= resolution[1]}
    """
    """
    fragments = [{'id': gop.id,
                  'source': gop.physical_id,
                  'start': gop.start_time,
                  'end': gop.end_time,
                  'fps': gop.fps}
             for physical in logical.videos()
             for gop in physical.gops()
             if physical.id in physical_videos and
                (t is None or
                 (gop.start_time <= t[0] and gop.end_time >= t[1]) or
                 (gop.start_time <= t[0] < gop.end_time) or
                 (gop.start_time < t[1] <= gop.end_time) or
                 (gop.start_time >= t[0] and gop.end_time <= t[1]))]
    """
    physical_videos = {physical_id:
        {'id': physical_id,
         'start': start_time,
         'end': end_time,
         'format': codec,
         'clock': 0,
         'size': size,
         'time': None,
         'resolution': (height, width),
         'frames': (end_time - start_time) * fps }
        for (physical_id, start_time, end_time, codec, height, width, size, fps) in VFS.instance().database.execute(
            'SELECT physical_videos.id, MIN(start_time), MAX(end_time), codec, height, width, SUM(size), MAX(fps) '
            'FROM gops, physical_videos '
            'WHERE logical_id = ? AND '
            '      height >= ? AND width >= ? AND '
            '      start_time < ? AND end_time >= ? AND '
            '      gops.physical_id = physical_videos.id '
            'GROUP BY physical_videos.id',
            (logical.id,
            resolution[0], resolution[1],
            t[1] if t is not None else 9999999,
            t[0] if t is not None else -1)).fetchall()
        #for physical in logical.videos()
        if t is None or
               (start_time is not None and end_time is not None and
                start_time < t[1] and end_time >= t[0]) and
               #(physical.start_time() or 0) <= t[0] and
               #(physical.end_time() or t[1]) >= t[1]) and
           #physical.start_time() is not None and
           height >= resolution[0] and
           width >= resolution[1]}

    fragments = [{'id': gop_id,
                  'source': physical_id,
                  'start': start_time,
                  'end': end_time,
                  'fps': fps}
             for (gop_id, physical_id, start_time, end_time, fps) in VFS.instance().database.execute(
                'SELECT gops.id, physical_id, start_time, end_time, fps '
                'FROM gops, physical_videos '
                'WHERE logical_id = ? AND '
                '      physical_id IN ({}) AND '
                '      gops.physical_id = physical_videos.id AND '
                '      (? = -1 OR ('
                '          (start_time <= ? AND end_time >= ?) OR '
                '          (start_time <= ? AND ? < end_time) OR '
                '          (start_time < ? AND ? <= end_time) OR '
                '          (start_time >= ? AND end_time <= ?)))'.format(','.join(map(str, physical_videos.keys()))),
                    (logical.id,
                     -1 if t is None else 0,
                     t[0] if t is not None else -1, t[1] if t is not None else -1,
                     t[0] if t is not None else -1, t[0] if t is not None else -1,
                     t[1] if t is not None else -1, t[1] if t is not None else -1,
                     t[0] if t is not None else -1, t[1] if t is not None else -1)).fetchall()]
    return physical_videos.values(), fragments

def solve_naive(logical, roi, t, fps, codec):
    videos = [video for video in logical.videos() if
              min(gop.start_time for gop in video.gops()) <= t[0] and
              max(gop.end_time for gop in video.gops()) >= t[1]]
    return [gop for gop in videos[-1].gops() if _between(gop, t)]

def solve_exact(logical, resolution, roi, t, fps, codec):
    #if roi is not None:
    #    return None

    gop_id = VFS.instance().database.execute("""
        SELECT gops.id FROM gops, physical_videos
        WHERE height = ? AND width = ? AND 
              start_time <= ? AND end_time >= ? AND 
              codec = ? AND
              gops.physical_id = physical_videos.id AND 
              logical_id = ?""", (resolution[0], resolution[1], t[0], t[1], codec, logical.id)).fetchone()

    return [Gop.get(gop_id[0])] if gop_id else None

def solve_constraint(logical, resolution, roi, t, fps, codec):
    physical, fragments = _prepare(logical, resolution, t)

    logging.info(f'Solving for {len(physical)} physical videos with {len(fragments)} fragments')

    if len(physical) == 1:
        # Only one physical video, so not really much to solve
        distinct_fragment_ids = (f['id'] for f in sorted(fragments, key=lambda f: f['start']) if f['end'] >= t[0] and f['start'] < t[1])
    elif len(physical) > 1:
        video_objects, goal_ints = build_from_video_info(physical, fragments, t)

        fragment_ids = find_best_intervals(video_objects, goal_ints, codec, resolution)
        distinct_fragment_ids = list(dict.fromkeys(fragment_ids))
    else:
        logging.error("No physical videos found for solver to examine")
        assert False

    gops = [Gop.get(id) for id in distinct_fragment_ids]
    return gops

def solve(logical, resolution, roi, t, fps, codec):
    #return solve_naive(logical, roi, t, fps, codec)
    with log_runtime('Solver'):
        return solve_exact(logical, resolution, roi, t, fps, codec) or \
               solve_constraint(logical, resolution, roi, t, fps, codec)