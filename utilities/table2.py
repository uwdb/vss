import os
import logging
import random
from concurrent.futures import ThreadPoolExecutor, wait

import cv2

from vfs.descriptor import Descriptor
from vfs.histogram import Histogram
from vfs.jointcompression import JointCompression

from vfs import engine, api
from vfs.gop import Gop
from vfs.logicalvideo import LogicalVideo
from vfs.utilities import log_runtime
from vfs.videoio import read_first_frame


# Copied from jointcompression.py
def execute():
    candidate_gop = JointCompression.get_candidate(epoch)

    if candidate_gop is None:
        return 0

    #VFS.instance().database.execute(
    #    'UPDATE gops SET examined = ? WHERE id = ?', (epoch + 1, candidate_gop.id)).close()

    closest_gops = Descriptor.closest_match(epoch, candidate_gop) if candidate_gop else (None, None)

    for gop_id, matches in closest_gops:
        overlap_gop = Gop.get(gop_id)

        if not overlap_gop is None:
            return JointCompression.co_compress(candidate_gop, overlap_gop, matches)
        else:
            logging.info("No GOP found")
            return None



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    n = 50
    mt = 0.0

    with engine.VFS(transient=True) as instance:
        api.vacuum()

        print(f'Joint compressed GOPs: {(instance.database.execute("SELECT SUM(joint) FROM gops").fetchone()[0] or 0) // 2}')

        if 'v3' not in api.list():
            api.write('v3', 'inputs/visualroad-2k-30a-gop30.mp4')
        if 'v2' not in api.list():
            api.write('v2', 'inputs/visualroad-2k-30b-gop30.mp4')

        for l in LogicalVideo.get_all():
            for v in l.videos():
                for g in v.gops():
                    if g.histogram is None:
                        logging.debug('MetadataWorker: creating metadata for GOP %d.%d', v.id, g.id)
                        engine._create_metadata_process(g.id, instance.database.filename, g.filename, v.height, v.width, v.codec)

        clusters = Histogram.cluster_all()
        epoch = 0

        while execute() is not None:
            pass
