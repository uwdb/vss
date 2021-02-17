import os
import logging
from vfs import api
from vfs import engine
from vfs.physicalvideo import PhysicalVideo
from vfs.rawcompression import compress
from vfs.videoio import encoded


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    t = (0, 12)
    level = 15
    #with engine.VFS(transient=True):
    #    api.write('v', 'inputs/visualroad-4k-30a.mp4')
    #    api.read('v', '/dev/null', t=t, codec='rgb')
        #os.remove('out.rgb')

    if level is not None:
        with engine.VFS(transient=True):
            for physical in PhysicalVideo.get_all():
                if not encoded[physical.codec]:
                    for gop in physical.gops():
                        if gop.zstandard != level:
                            compress(gop.id, level)

    with engine.VFS(transient=True):
        api.read('v', '/dev/null', t=t, codec='rgb')
