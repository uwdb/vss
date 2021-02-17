import os
import logging
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.pool import Pool, ThreadPool

from vfs.engine import VFS
from vfs.logicalvideo import LogicalVideo
from vfs.physicalvideo import PhysicalVideo
import vfs.reconstruction
import vfs.solver
from vfs.videoio import H264, encoded
from vfs.utilities import log_runtime

pools = {}

def write(name, filename, resolution=None, codec=None, fps=None, budget=None):
    if not LogicalVideo.exists_by_name(name):
        with log_runtime('API: write'):
            logical = LogicalVideo.add(name)
            physical = PhysicalVideo.load(logical, filename, resolution, codec, fps)
            logical.budget = physical.size() * (budget or
                                                (logical.DEFAULT_ENCODED_BUDGET_MULTIPLIER if encoded[physical.codec] else
                                                 logical.DEFAULT_RAW_BUDGET_MULTIPLIER))
    else:
        print("ERROR: video already exists")

def read(name, filename=None, resolution=None, roi=None, t=None, codec=None, fps=None, child_process=False):
    #if isinstance(name_or_tuple, tuple):
    #    name, filename, resolution, roi, t, codec, fps, child_process = name_or_tuple
    #else:
    #    name = name_or_tuple
    if t is not None and t[1] - t[0] < 1:
        vfs.reconstruction.POOL_SIZE = 1

    if LogicalVideo.exists_by_name(name):
        with log_runtime('API: read', level=logging.ERROR):
            logical = LogicalVideo.get_by_name(name)

            t = tuple(map(float, t)) if t is not None else (0, logical.duration())
            resolution = resolution or ((roi[2] - roi[0], roi[3] - roi[1]) if roi else next(logical.videos()).resolution())
            codec = codec or H264

            gops = vfs.solver.solve(logical, resolution, roi, t, fps, codec)

            if filename is not None:
                vfs.reconstruction.reconstruct(filename, logical, gops, resolution, roi, t, fps, codec)
            else:
                filename = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex + ".mp4")
                result_filename = vfs.reconstruction.reconstruct(filename, logical, gops, resolution, roi, t, fps, codec, is_stream=True)
                return open(result_filename, 'rb') if not child_process else result_filename
    else:
        print("ERROR: video does not exist")

def readmany(names, filenames=None, resolutions=None, rois=None, ts=None, codecs=None, fpss=None, workers=None):
    global pools

    futures = []
    workers = workers or 4
    if workers not in pools:
        pools[workers] = ProcessPoolExecutor(max_workers=workers)
    pool = pools[workers]

    #with ProcessPoolExecutor(max_workers=workers or 4) as pool:
    for name, filename, resolution, roi, t, codec, fps in zip(names, filenames, resolutions, rois, ts, codecs, fpss):
        futures.append(pool.submit(read, name, filename, resolution, roi, t, codec, fps, child_process=filename is None))
    return (f.result() for f in futures)

def readmany2(names, filenames=None, resolutions=None, rois=None, ts=None, codecs=None, fpss=None, workers=None):
    global pools

    workers = workers or 4
    if workers not in pools:
        pools[workers] = ProcessPoolExecutor(max_workers=workers)
    pool = pools[workers]

    #with ProcessPoolExecutor(max_workers=workers or 4) as pool:
    for name, filename, resolution, roi, t, codec, fps in zip(names, filenames, resolutions, rois, ts, codecs, fpss):
        yield pool.submit(read, name, filename, resolution, roi, t, codec, fps, child_process=filename is None)

def readmany2i(names, filenames=None, resolutions=None, rois=None, ts=None, codecs=None, fpss=None, workers=None):
    global pools

    workers = workers or 4
    if workers not in pools:
        pools[workers] = ProcessPoolExecutor(max_workers=workers)
    pool = pools[workers]

    #with ProcessPoolExecutor(max_workers=workers or 4) as pool:
    for index, (name, filename, resolution, roi, t, codec, fps) in enumerate(zip(names, filenames, resolutions, rois, ts, codecs, fpss)):
        future = pool.submit(read, name, filename, resolution, roi, t, codec, fps, child_process=filename is None)
        future.index = index
        yield future

#def readmany2(names, filenames=None, resolutions=None, rois=None, ts=None, codecs=None, fpss=None, workers=None):
#    futures = []
#    with ProcessPoolExecutor(max_workers=workers or 4) as pool:
#        return pool.map(read, zip(names, filenames, resolutions, rois, ts, codecs, fpss, [True] * len(names)))
#        for name, filename, resolution, roi, t, codec, fps in zip(names, filenames, resolutions, rois, ts, codecs, fpss):
#            yield pool.submit(read, name, filename, resolution, roi, t, codec, fps, child_process=filename is None)

def preadmany(reads, workers=None, pool=None):
    if pool is None:
        with ThreadPool(workers or 4) as pool:
            return pool.starmap(read, reads)
    else:
        return pool.starmap(read, reads)

def list():
    with log_runtime('API: list'):
        return [logical.name for logical in LogicalVideo.get_all()]

def delete(name):
    if LogicalVideo.exists_by_name(name):
        with log_runtime('API: delete'):
            LogicalVideo.delete(LogicalVideo.get_by_name(name))
    else:
        print("ERROR: video does not exist")

def start():
    return VFS()

def vacuum():
    # Clean broken physical videos
    for p in PhysicalVideo.get_all():
        if not any(p.gops()):
            print(f'Vacuum {p.id}')
            PhysicalVideo.delete(p)

        for g in p.gops():
            if (not os.path.exists(g.filename) or os.path.getsize(g.filename) == 0) and not g.joint:
                print(f'Vacuum {p.id}.{g.id} {g.filename}')
                VFS.instance().database.execute("DELETE FROM gops WHERE physical_id = ?", p.id)
                VFS.instance().database.execute("DELETE FROM physical_videos WHERE id = ?", p.id)

