from vfs.engine import VFS
from vfs.logicalvideo import LogicalVideo
from vfs.physicalvideo import PhysicalVideo
import vfs.reconstruction
import vfs.solver
from vfs.videoio import H264, encoded


def write(name, filename, resolution=None, codec=None, fps=None, budget=None):
    if not LogicalVideo.exists_by_name(name):
        logical = LogicalVideo.add(name)
        physical = PhysicalVideo.load(logical, filename, resolution, codec, fps)
        logical.budget = physical.size() * (budget or
                                            (logical.DEFAULT_ENCODED_BUDGET_MULTIPLIER if encoded[physical.codec] else
                                             logical.DEFAULT_RAW_BUDGET_MULTIPLIER))
    else:
        print("ERROR: video already exists")

def read(name, filename, resolution=None, roi=None, t=None, codec=None, fps=None):
    if LogicalVideo.exists_by_name(name):
        logical = LogicalVideo.get_by_name(name)

        t = tuple(map(float, t)) if t is not None else (0, logical.duration())
        resolution = resolution or next(logical.videos()).resolution()
        codec = codec or H264

        gops = vfs.solver.solve(logical, roi, t, fps, codec)
        vfs.reconstruction.reconstruct(filename, logical, gops, resolution, roi, t, fps, codec)
    else:
        print("ERROR: video does not exist")

def list():
    for video in LogicalVideo.get_all():
        print(video.name)

def delete(name):
    if LogicalVideo.exists_by_name(name):
        LogicalVideo.delete(LogicalVideo.get_by_name(name))
    else:
        print("ERROR: video does not exist")

def start():
    return VFS()
