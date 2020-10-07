import numpy as np
from vfs.videoio import HEVC, H264, RGB8
from vfs.gop import Gop
from vfs.constraints import build_from_video_info, find_best_intervals

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

def _prepare(logical, t):
    physical = [{'id': physical.id,
                              'start': physical.start_time(),
                              'end': physical.end_time(),
                              'format': physical.codec,
                              'clock': 0,
                              'size': physical.size(),
                              'time': None,
                              'frames': sum(gop.fps for gop in physical.gops()) }
                for physical in logical.videos()
                if t is None or ((physical.start_time() or 0) <= t[0] and
                                 (physical.end_time() or t[1]) >= t[1]) and
                   physical.start_time() is not None]
    fragments = [{'id': gop.id,
                  'source': gop.physical_id,
                  'start': gop.start_time,
                  'end': gop.end_time,
                  'fps': gop.fps}
                 for physical in logical.videos()
                 for gop in physical.gops()]
    return physical, fragments

def solve_naive(logical, roi, t, fps, codec):
    videos = [video for video in logical.videos() if
              min(gop.start_time for gop in video.gops()) <= t[0] and
              max(gop.end_time for gop in video.gops()) >= t[1]]
    return [gop for gop in videos[-1].gops() if _between(gop, t)]

def solve_constraint(logical, roi, t, fps, codec):
    physical, fragments = _prepare(logical, t)
    video_objects, goal_ints = build_from_video_info(physical, fragments, t)
    goal = codec
    fragment_ids = find_best_intervals(video_objects, goal_ints, goal)
    gops = [Gop.get(id) for id in fragment_ids]
    return gops

def solve(logical, roi, t, fps, codec):
    #return solve_naive(logical, roi, t, fps, codec)
    return solve_constraint(logical, roi, t, fps, codec)