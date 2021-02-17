import os
import logging
import random
import cv2

from vfs.descriptor import Descriptor
from vfs.jointcompression import JointCompression

from vfs import engine, api
from vfs.gop import Gop
from vfs.utilities import log_runtime
from vfs.videoio import read_first_frame


def get_random_gop(engine, name, r, t, index, avoid_ids=None):
    gops = engine.database.execute("SELECT gops.id FROM gops, physical_videos, logical_videos WHERE logical_videos.id == physical_videos.logical_id AND gops.physical_id = physical_videos.id AND height = ? AND width = ? AND name = ? ORDER BY gops.id LIMIT ?", (r[0], r[1], name, index + 1)).fetchall()
    #gops = engine.database.execute("SELECT gops.id FROM gops, physical_videos, logical_videos WHERE logical_videos.id == physical_videos.logical_id AND gops.physical_id = physical_videos.id AND height = ? AND width = ? AND end_time - start_time != ? AND name = ?", (r[0], r[1], t, name)).fetchall()
    gop = Gop.get(gops[-1]) #random.choice(gops))

    if not os.path.exists(gop.filename):
        return get_random_gop(engine, name, r, t, index, avoid_ids)
    elif gop.id in (avoid_ids or []):
        return get_random_gop(engine, name, r, t, index, avoid_ids)
    else:
        return gop

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    n = 50
    mt = 0.0

    with engine.VFS(transient=True) as engine:
        api.vacuum()

        #api.write('v1', 'inputs/visualroad-4k-30a-gop30.mp4')
        #api.write('v2', 'inputs/visualroad-4k-30b-gop30.mp4')
        #api.read('v1', 'out.mp4', resolution=(1080, 1920), t=(0, 60))
        #api.read('v2', 'out.mp4', resolution=(1080, 1920), t=(0, 60))
        #api.read('v1', 'out.mp4', resolution=(540, 960), t=(0, 60))
        #api.read('v2', 'out.mp4', resolution=(540, 960), t=(0, 60))

        for i in range(n):
            r = 540, 960 # 1080, 1920 #2160, 3840 # # #
            gop1 = get_random_gop(engine, 'v1', r, 1, i)
            gop2 = get_random_gop(engine, 'v2', r, 1, i, [gop1.id])

            # Compression
            #with log_runtime('Compression:'):
            frame1 = read_first_frame(gop1.filename)
            frame2 = read_first_frame(gop2.filename)

            #cuMat1 = cv2.cuda_GpuMat()
            #cuMat1.upload(frame)

            # Feature Detection
            #with log_runtime('Feature detection:'):
            keypoints1, descriptors1 = Descriptor.create(frame1)
            keypoints2, descriptors2 = Descriptor.create(frame2)
            engine.database.execute(
                'UPDATE gops SET keypoints = ?, descriptors = ? WHERE id = ?',
                (keypoints1, descriptors1, gop1.id))
            engine.database.execute(
                'UPDATE gops SET keypoints = ?, descriptors = ? WHERE id = ?',
                (keypoints2, descriptors2, gop2.id))
            gop1.keypoints, gop2.keypoints = keypoints1, keypoints2
            gop1.descriptors, gop2.descriptors = descriptors1, descriptors2

            # Homography
            #with log_runtime('Homography:'):
            success, matches = Descriptor.adhoc_match(descriptors1, descriptors2)
            JointCompression.estimate_homography(gop1, gop2, matches)

            # Joint compression
            with log_runtime('Joint:') as t:
                JointCompression.co_compress(gop1, gop2, matches, abort_psnr_threshold=0, dryrun=True)

            mt += t.duration

            #for _ in range(50):
            #with log_runtime('Estimation:'):
            #     JointCompression.estimate_homography(gop1, gop2)
            #print(1)
        print(mt / n)