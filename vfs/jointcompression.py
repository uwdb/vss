import os
from os import path
import threading
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cv2
#from skimage.measure import compare_ssim
from skimage import measure

from vfs.videoio import VideoReader, VideoWriter, NullReader, NullWriter, read_first_frame
from vfs.engine import VFS
from vfs.physicalvideo import PhysicalVideo
from vfs.descriptor import Descriptor
import vfs.homography
import vfs.utilities
from vfs.utilities import roundeven, log_runtime

class JointCompression(object):
    LEFT = 'left'
    RIGHT = 'right'
    OVERLAP = 'overlap'

    _homography_pool = ThreadPoolExecutor(max_workers=4)

    @classmethod
    def get_candidate(cls, epoch):
        from vfs.gop import Gop

        smallest_cluster = (VFS.instance().database.execute(
            'SELECT cluster_id, COUNT(*) FROM gops candidate '
              'WHERE examined <= ? AND joint = 0 AND '
              '      cluster_id > 0 '
              # What did this do?
              #'      physical_id < (SELECT MAX(physical_id) FROM gops WHERE examined <= ? AND joint = 0 AND cluster_id = candidate.cluster_id) '
              'GROUP BY cluster_id '
              'ORDER BY COUNT(*) ASC '
              'LIMIT 1', (epoch)).fetchone() or [None])[0] #(epoch, epoch)

        if smallest_cluster is not None:
            return Gop.get(*VFS.instance().database.execute(
                'SELECT MIN(id) FROM gops '
                'WHERE cluster_id = ? AND examined <= ? AND joint = 0',
                (smallest_cluster, epoch)).fetchone())
        else:
            return None

    lock = threading.Lock()

    @classmethod
    def execute(cls, epoch, pool):
        candidate_gop = JointCompression.get_candidate(epoch)
        #overlap_gop, matches = Descriptor.closest_match(epoch, candidate_gop) if candidate_gop else (None, None)
        #closest_gops = Descriptor.closest_match(epoch, candidate_gop) if candidate_gop else (None, None)
        if candidate_gop is None: # or closest_gops == []:
        #if candidate_gop is None or overlap_gop is None:
                return False, None, None

        VFS.instance().database.execute(
            'UPDATE gops SET examined = ? WHERE id = ?', (epoch + 1, candidate_gop.id)).close()

        def task(candidate_gop):
            from vfs.gop import Gop

            closest_gops = Descriptor.closest_match(epoch, candidate_gop) if candidate_gop else (None, None)
            overlap_gop = None

            with cls.lock:
                for gop_id, matches in closest_gops:
                    if VFS.instance().database.execute(
                        'SELECT examined FROM gops WHERE id = ? AND joint = 0', gop_id).fetchone()[0] <= epoch:
                        overlap_gop = Gop.get(gop_id)
                        VFS.instance().database.execute(
                            'UPDATE gops SET examined = ? WHERE id = ?', (epoch + 1, overlap_gop.id)).close()
                        break

            if not overlap_gop is None:
                #VFS.instance().database.execute(
                #    'UPDATE gops SET examined = ? WHERE id in (?, ?)', (epoch + 10, candidate_gop.id, overlap_gop.id)).close()
                #threading.Thread(target=cls.co_compress, args=(candidate_gop, overlap_gop, matches)).start()
                return cls.co_compress(candidate_gop, overlap_gop, matches)
                #return True, candidate_gop, pool.submit(cls.co_compress, candidate_gop, overlap_gop, matches)
                #return True, candidate_gop, future #bytes_saved
            elif not candidate_gop is None:
                logging.info("Deferring joint compression for gop %d-%d", candidate_gop.physical_id, candidate_gop.id)
                VFS.instance().database.execute(
                    'UPDATE gops SET examined = ? WHERE id = ?', (epoch + 1, candidate_gop.id)).close()
                return 0 #False, candidate_gop, None
            else:
                return 0 #False, None, None

        return True, candidate_gop, pool.submit(task, candidate_gop)

    #output = open('joint.csv', "w")

    @classmethod
    def estimate_homography(cls, gop1, gop2, frame1=None, frame2=None, matches=None, fast=True):
        if matches is None:
            frame1 = frame1 if frame1 is not None else read_first_frame(gop1.filename)
            #with VideoReader(gop1.filename, shape=gop1.video().shape(), codec=gop1.video().codec, limit=1) as reader1:
            #    frame1 = reader1.read()
            frame2 = frame2 if frame2 is not None else read_first_frame(gop2.filename)
            #with VideoReader(gop2.filename, shape=gop1.video().shape(), codec=gop2.video().codec, limit=1) as reader2:
            #    frame2 = reader2.read()
            keypoints1, descriptors1 = Descriptor.create(frame1, fast=fast)
            keypoints2, descriptors2 = Descriptor.create(frame2, fast=fast)

            has_homography, matches = Descriptor.adhoc_match(descriptors1, descriptors2, fast=fast)
            #keypoints1, keypoints2 = keypoints1, keypoints2
        else:
            keypoints1, keypoints2 = gop1.keypoints, gop2.keypoints
            has_homography = True

        with log_runtime('Homography:'):
            H, Hi = vfs.homography.project(keypoints1, keypoints2, matches)

        frame2_overlap_yoffset = -int(round(np.dot([0, gop1.video().height, 1], Hi)[0]))
        #frame2_overlap_yoffset = max(-int(round(np.dot([0, gop1.video().height, 1], Hi)[0])), 0)
        frame1_left_width = roundeven(Hi.dot([0,0,1])[0])
        frame2_right_width = max(roundeven(gop2.video().width - H.dot([gop1.video().width, 0, 1])[0] / H.dot([gop1.video().width, 0, 1])[2]), 0)
        overlap_height = gop2.video().height + 2 * frame2_overlap_yoffset
        overlap_width = gop1.video().width - frame1_left_width

        if frame1_left_width < 0:
            return H, Hi, cls._create_image(height=0, width=0), cls._create_image(height=0, width=0), None, None, None, None, True, has_homography

        if frame1_left_width < 0:
            frame1_left_width = 0
        if overlap_width < 0:
            overlap_width = 0
        if overlap_height < 0:
            overlap_height = 0

        pretransform_points = np.float32([
            [frame1_left_width, 0],
            [gop1.video().width, 0],
            [frame1_left_width, gop1.video().height],
            [gop1.video().width, gop1.video().height]])
        posttransform_points = np.float32(
            [[0, 0 + frame2_overlap_yoffset],
             [gop1.video().width - frame1_left_width + 1, 0 + frame2_overlap_yoffset],
             [0, gop1.video().height + frame2_overlap_yoffset],
             [gop1.video().width - frame1_left_width + 1, gop1.video().height + frame2_overlap_yoffset]])

        transform = cv2.getPerspectiveTransform(pretransform_points, posttransform_points)
        inverse_transform = cv2.getPerspectiveTransform(posttransform_points, pretransform_points)
        # Hi = transform.dot(Hi)
        Ho = H
        H, Hi = H.dot(inverse_transform), transform.dot(Hi)

        logging.info('Frames left %d, right %d, overlap %d x %d' % (frame1_left_width, frame2_right_width, overlap_height, overlap_width))
        left = cls._create_image(gop1.video(), width=frame1_left_width)
        right = cls._create_image(gop2.video(), width=frame2_right_width)
        overlap = cls._create_image(height=overlap_height, width=overlap_width)
        overlap_subframe = cls._create_image(height=gop1.video().height, width=overlap_width, dtype=np.uint16)
        recovered_frame2 = np.empty(gop2.video().shape(), dtype=np.uint8)

        return H, Hi, left, right, overlap, overlap_subframe, recovered_frame2, frame2_overlap_yoffset, False, has_homography

    @classmethod
    def mean_join(cls, overlap, overlap_subframe, left):
        np.copyto(overlap_subframe, overlap)
        nonzeros = np.sum(overlap_subframe, axis=2) >= 9
        overlap_subframe = overlap_subframe + left
        overlap_subframe[nonzeros] //= 2
        np.copyto(overlap, overlap_subframe)

        #np.copyto(overlap_subframe, overlap)
        #overlap_subframe = (overlap_subframe + left) // 2
        #np.copyto(overlap, overlap_subframe)

    @classmethod
    def left_join(cls, overlap, left):
        np.copyto(overlap, left)

    @classmethod
    def blend_join(cls, output, right, left):
        alpha_left = np.empty_like(left)
        alpha_left[:] = np.linspace(0.0, 1.0, num=left.shape[1], dtype=float)

        alpha_right = np.empty_like(right)
        alpha_right[:] = np.linspace(0.0, 1.0, num=right.shape[1], dtype=float)

        cv2.addWeighted(left, alpha_left, right, alpha_right, 0, dst=output, dtype=np.uint8)

        #def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
#             axis=0):

        distance_left = cv2.distanceTransform(border_left, cv2.DIST_L2, 3)


        left_mask = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_mask = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        cv2.threshold(left_mask, left_mask, 255, 255, cv2.THRESH_BINARY_INV or cv2.THRESH_OTSU)
        cv2.threshold(right_mask, right_mask, 255, 255, cv2.THRESH_BINARY_INV or cv2.THRESH_OTSU)

        out = cls.computeAlphaBlending(left, left_mask, right, right_mask)

    @classmethod
    def alpha_blend(cls, left, left_mask, right, right_mask):
        both_masks = left_mask or right_mask
        #cv::imshow("maskOR", bothMasks);
        no_mask = 255 - both_masks

        raw_alpha = cls._create_image(template=no_mask, dtype=float)
        raw_alpha = 1.0

        border_left = 255 - cls.border(left_mask)
        border_right = 255 - cls.border(right_mask)

        distance_left = cv2.distanceTransform(border_left, cv2.DIST_L2, 3)
        _, max, _ = cv2.minMaxLoc(distance_left, mask=left_mask[distance_left > 1])
        distance_left = distance_left * 1.0 / max

        distance_right = cv2.distanceTransform(border_right, cv2.DIST_L2, 3)
        _, max, _ = cv2.minMaxLoc(distance_right, mask=right_mask[distance_right > 1])
        distance_right = distance_left * 1.0 / max

        cv2.copyTo(raw_alpha, left_mask, distance_left)

        """
        rawAlpha.copyTo(dist1Masked,noMask);    // edited: where no mask is set, blend with equal values
        dist1.copyTo(dist1Masked,mask1);
        rawAlpha.copyTo(dist1Masked,mask1&(255-mask2)); //edited
        """

    @classmethod
    def interlace_join(cls):
        # TODO support interlace mode
        raise NotImplemented()

    @classmethod
    def recovered_right_psnr(cls, H, overlap, frame2, recovered_frame2, right):
        cv2.warpPerspective(overlap, H, dsize=tuple(reversed(recovered_frame2.shape[:2])), dst=recovered_frame2)
        np.copyto(recovered_frame2[:, -right.shape[1]:], frame2[:, -right.shape[1]:])
        psnr = vfs.utilities.psnr(frame2, recovered_frame2)
        return psnr

    @classmethod
    def recovered_left_psnr(cls, overlap, frame1, left):
        recovered_frame1 = np.hstack([left, overlap])
        psnr = vfs.utilities.psnr(frame1, recovered_frame1)
        #cv2.imwrite('left.png', frame1)
        #cv2.imwrite('rleft.png', recovered_frame1)
        #cv2.imwrite('overlap.png', overlap)
        return psnr

    @classmethod
    def co_compress(cls, gop1, gop2, matches, abort_psnr_threshold=25, dryrun=False, gops_reversed=False):
        #cls.output.write('%s,%d,%d,%s,%d,%d\n' % (gop1.video().logical().name, gop1.video().id, gop1.id,
        #             gop2.video().logical().name, gop2.video().id, gop2.id))
        #cls.output.flush()
        logging.info('Joint compress %s-%d-%d and %s-%d-%d (%d matches, %s, %s)',
                     gop1.video().logical().name, gop1.video().id, gop1.id,
                     gop2.video().logical().name, gop2.video().id, gop2.id,
                     len(matches),
                     gop1.filename, gop2.filename)

        assert(gop1.id != gop2.id)
        assert(gop1.video().codec == gop2.video().codec)
        assert(not gop1.joint)
        assert(not gop2.joint)

        H, Hi, left, right, overlap, overlap_subframe, recovered_frame2, frame2_overlap_yoffset, inverted_homography, has_homography = cls.estimate_homography(gop1, gop2, matches=None, fast=False) #matches) #TODO fast, matches
        Hs = np.hstack([[0], H.flatten()])

        #H, Hi = homography.project(gop1.keypoints, gop2.keypoints, matches)

        #frame2_overlap_yoffset = -int(round(np.dot([0, gop1.video().height, 1], Hi)[0]))
        #frame1_left_width = roundeven(Hi.dot([0,0,1])[0])
        #frame2_right_width = roundeven(gop2.video().width - H.dot([gop1.video().width, 0, 1])[0] / H.dot([gop1.video().width, 0, 1])[2])
        #overlap_height = gop2.video().height + 2 * frame2_overlap_yoffset
        #overlap_width = gop1.video().width - frame1_left_width

        if has_homography and inverted_homography and not gops_reversed:
            return cls.co_compress(gop2, gop1, matches, gops_reversed=True)
        # Are videos identical?
        elif not inverted_homography and left.shape[1] == 0 and right.shape[1] == 0:
        #if frame1_left_width == 0 and frame2_right_width == 0:
            return cls.deduplicate(gop1, gop2)
        # Are left/right frames too small to encode?
        elif 0 <= left.shape[1] < 32 or 0 <= right.shape[1] < 32 or 0 <= overlap.shape[1] < 32:
            logging.info('Joint compression aborted; left/right/overlap frames too small (%d, %d)', gop1.id, gop2.id)
            VFS.instance().database.execute(
                'UPDATE gops SET examined=9999998 '  # 9999998 = possibly examine again?
                'WHERE id in (?, ?)',
                (gop1.id, gop2.id)).close()
            return 0
        #    return cls.deduplicate(gop1, gop2)
        else:
            #pretransform_points = np.float32([
            #    [frame1_left_width, 0],
            #    [gop1.video().width, 0],
            #    [frame1_left_width, gop1.video().height],
            #    [gop1.video().width, gop1.video().height]])
            #posttransform_points = np.float32(
            #    [[0, 0 + frame2_overlap_yoffset],
            #    [gop1.video().width - frame1_left_width + 1, 0 + frame2_overlap_yoffset],
            #    [0, gop1.video().height + frame2_overlap_yoffset],
            #    [gop1.video().width - frame1_left_width + 1, gop1.video().height + frame2_overlap_yoffset]])

            #transform = cv2.getPerspectiveTransform(pretransform_points, posttransform_points)
            #inverse_transform = cv2.getPerspectiveTransform(posttransform_points, pretransform_points)
            ##Hi = transform.dot(Hi)
            #Ho = H
            #H, Hi = H.dot(inverse_transform), transform.dot(Hi)

            #left = cls._create_image(gop1.video(), width=frame1_left_width)
            #right = cls._create_image(gop2.video(), width=frame2_right_width)
            #overlap = cls._create_image(height=overlap_height, width=overlap_width)
            #recovered_frame2 = np.empty(gop2.video().shape(), dtype=np.uint8)

            filenametemplate = '{}-{{}}{}'.format(*path.splitext(gop1.filename))
            abort = False
            frame_index = 0
            total_left_psnr, total_right_psnr = 0, 0
            codec = gop1.video().codec

            # if frame1_left_width else NullWriter() as leftwriter, \
            #if frame2_right_width else NullWriter() as rightwriter, \
            with log_runtime('Joint compression:'):
                with VideoReader(gop1.filename, gop1.video().shape(), codec) as reader1, \
                     VideoReader(gop2.filename, gop2.video().shape(), codec) as reader2, \
                     VideoWriter(filenametemplate.format(cls.LEFT), left.shape, codec) \
                             if left.shape[1] else NullWriter() as leftwriter, \
                     VideoWriter(filenametemplate.format(cls.RIGHT), right.shape, codec) \
                             if right.shape[1] else NullWriter() as rightwriter, \
                     VideoWriter(filenametemplate.format(cls.OVERLAP), overlap.shape, codec) as overlapwriter:
                    while (not reader1.eof or not reader2.eof) and not abort:
                        attempts = 0
                        frame_index += 1
                        frame1, frame2 = reader1.read(), reader2.read()

                        while attempts < 2:
                            attempts += 1

                            if frame1 is not None and frame2 is not None:
                                pass
                            elif frame1 is not None and frame2 is None:
                                frame2 = np.zeros(gop2.video().shape(), dtype=np.uint8)
                            elif frame2 is not None and frame1 is None:
                                frame1 = np.zeros(gop1.video().shape(), dtype=np.uint8)

                            if frame1 is not None or frame2 is not None:
                                # Create and write overlap
                                cv2.warpPerspective(frame2, Hi, dsize=tuple(reversed(overlap.shape[:2])), dst=overlap)

                                # Left join
                                """cv2.imwrite('frame1.png', frame1)
                                cv2.imwrite('frame2.png', frame2)
                                cv2.imwrite('overlap.png', overlap)
                                print(gop1.filename)
                                print(gop2.filename)
                                print(gop1.video().shape())
                                print(left.shape)
                                print(right.shape)
                                print(overlap.shape)
                                print(frame2_overlap_yoffset, frame2_overlap_yoffset + gop1.video().height)
                                print(left.shape[1])
                                print(Hi)"""
                                #tmp = frame2_overlap_yoffset
                                #if frame2_overlap_yoffset < 0:
                                #    frame2_overlap_yoffset = 0

                                # Mean join
                                #cls.mean_join(overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height], overlap_subframe, frame1[:, left.shape[1]:])
                                #np.copyto(overlap_subframe, overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height])
                                #overlap_subframe = (overlap_subframe + frame1[:, left.shape[1]:]) // 2
                                #np.copyto(overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height], overlap_subframe)

                                #mean_subframe = np.copy(overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height]).astype(np.uint16) // 2
                                ##mean_subframe = np.copy(overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height]).astype(np.uint16)
                                #mean_subframe = (mean_subframe + frame1[:, left.shape[1]:]) // 2
                                ##mean_subframe //= 2
                                #np.copyto(overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height],
                                #          mean_subframe.astype(np.uint8))

                                # Left join
                                cls.left_join(overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height], frame1[:, left.shape[1]:])
                                #np.copyto(overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height],
                                #          frame1[:, left.shape[1]:])

                                #frame2_overlap_yoffset = tmp
                                # Mean join (has a bug in non-overlapping left)
                                #overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height] //= 2
                                #overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height] += frame1[:, left.shape[1]:] // 2

                                if left.shape[1] != 0:
                                    np.copyto(left, frame1[:, :left.shape[1]])
                                if right.shape[1] != 0:
                                    np.copyto(right, frame2[:, -right.shape[1]:])

                                right_psnr = cls.recovered_right_psnr(H, overlap, frame2, recovered_frame2, right)
                                left_psnr = cls.recovered_left_psnr(overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height], frame1, left)

                                #cv2.warpPerspective(overlap, H, dsize=tuple(reversed(recovered_frame2.shape[:2])), dst=recovered_frame2)
                                #np.copyto(recovered_frame2[:, -right.shape[1]:], frame2[:, -right.shape[1]:])
                                #psnr = vfs.utilities.psnr(frame2, recovered_frame2)

                                if right_psnr < abort_psnr_threshold:
                                    if attempts == 2:
                                        abort = True
                                        break
                                    else:
                                        logging.debug(f"Recomputing homography ({gop1.id} <-> {gop2.id}) PSNR {right_psnr:0.1f}")
                                        #TODO save new homography
                                        H, Hi, left, right, overlap, overlap_subframe, recovered_frame2, frame2_overlap_yoffset, inverted_homography, has_homography = cls.estimate_homography(
                                            gop1, gop2, frame1=frame1, frame2=frame2, matches=None, fast=True) #matches)
                                        np.vstack([Hs, np.hstack([[frame_index], H.flatten()])])
                                        #Hs.append(np.hstack([[frame_index], H.flatten()]))
                                else:
                                    total_right_psnr += right_psnr
                                    total_left_psnr += left_psnr
                                    attempts = 999
                                    #break

                                    leftwriter.write(left)
                                    rightwriter.write(right)
                                    overlapwriter.write(overlap)

            if abort:
                #cv2.imwrite('abortframe1_%d.png' % gop1.id, frame1)
                #cv2.imwrite('abortframe2_%d.png' % gop1.id, frame2)
                #cv2.imwrite('abortleft_%d.png' % gop1.id, left)
                #cv2.imwrite('abortright_%d.png' % gop1.id, right)
                cv2.imwrite('abortoverlap.png', overlap)
                cv2.imwrite('abortrecovered1.png', np.hstack([left, overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height]]))
                cv2.imwrite('abortrecovered2.png', recovered_frame2)
                logging.info('Joint compression aborted; quality threshold violated %d < %d (%d vs %d)',
                             right_psnr, abort_psnr_threshold, gop1.id, gop2.id)
                #ssim = compare_ssim(frame2, recovered_frame2, multichannel=True)
                os.remove(filenametemplate.format(cls.LEFT))
                os.remove(filenametemplate.format(cls.OVERLAP))
                os.remove(filenametemplate.format(cls.RIGHT))
                VFS.instance().database.executebatch([
                    f'INSERT INTO gop_joint_aborted(gop1, gop2) VALUES ({gop1.id}, {gop2.id})',
                    f'INSERT INTO gop_joint_aborted(gop1, gop2) VALUES ({gop2.id}, {gop1.id})',
                    f'UPDATE gops SET examined=examined + 1 WHERE id in ({gop1.id}, {gop2.id})'])
                return 0
            elif not dryrun:
                    original_size = path.getsize(gop1.filename) + path.getsize(gop2.filename)

                    VFS.instance().database.execute(
                        'UPDATE gops SET examined=9999999, joint=1, original_filename=filename, filename=?, homography=?, shapes=?, is_left=(id=?) '
                               'WHERE id in (?, ?)',
                               (filenametemplate, np.vstack(Hs), np.vstack([left.shape, overlap.shape, right.shape]),
                                gop1.id, gop1.id, gop2.id)).close()

                    os.remove(gop1.filename)
                    os.remove(gop2.filename)

                    bytes_saved = (original_size -
                                 (path.getsize(filenametemplate.format(cls.LEFT)) +
                                  path.getsize(filenametemplate.format(cls.OVERLAP)) +
                                  path.getsize(filenametemplate.format(cls.RIGHT))))
                    logging.info('Joint compression saved %dKB (%d%%), %d frames, PSNR left=%d, right=%d', bytes_saved // 1000, (bytes_saved * 100) // original_size, frame_index-1, total_left_psnr // (frame_index-1), total_right_psnr // (frame_index-1))
                    with open('joint.csv', 'a') as f:
                        f.write(f'{gop1.id},{gop2.id},{frame_index-1},{total_right_psnr // (frame_index-1)},{total_left_psnr // (frame_index-1)}\n')
                    return bytes_saved
            else:
                return 0

    @classmethod
    def co_decompress(cls, gop, filename):
        logging.info('Joint decompress %s-%d-%d',
                     gop.video().logical().name, gop.video().id, gop.id)

        assert(gop.joint)

        left_filename = gop.filename.format(cls.LEFT)
        right_filename = gop.filename.format(cls.RIGHT)
        overlap_filename = gop.filename.format(cls.OVERLAP)
        codec = gop.video().codec

        H, shapes, is_left = VFS.instance().database.execute(
            'SELECT homography, shapes, is_left FROM gops WHERE id = ?', gop.id).fetchone()
        left_shape, overlap_shape, right_shape = shapes[0], shapes[1], shapes[2]
        frame = np.zeros(gop.video().shape(), dtype=np.uint8)

        if H.shape[1] > 1:
            # Current code assumes H=3x3 matrix, but actually list of hstack[frameid, 3x3.flatten] matrices
            raise NotImplementedError("Need to add support for reprojection")
        else:
            H = H[1:, 0].reshape((3, 3))

        with VideoReader(left_filename, left_shape, codec) if is_left else NullReader() as leftreader, \
             VideoReader(overlap_filename, overlap_shape, codec) as overlapreader, \
             VideoReader(right_filename, right_shape, codec) if not is_left else NullReader() as rightreader, \
             VideoWriter(filename, gop.video().shape(), codec) as writer:
            while not leftreader.eof or not overlapreader.eof or not rightreader.eof:
                if is_left:
                    left, overlap = leftreader.read(), overlapreader.read()

                    if overlap is not None:
                        top = overlap.shape[0]//2 - frame.shape[0]//2
                        bottom = top + frame.shape[0]
                        np.copyto(frame[:, left.shape[1]:], overlap[top:bottom])

                    if left is not None:
                        np.copyto(frame[:, :left.shape[1]], left)
                else:
                    overlap, right = overlapreader.read(), rightreader.read()

                    if overlap is not None:
                        cv2.warpPerspective(overlap, H, dsize=tuple(reversed(frame.shape[:2])), dst=frame)

                    if right is not None:
                        np.copyto(frame[:, -right.shape[1]:], right)

                writer.write(frame)
        pass

    @classmethod
    def deduplicate(cls, gop1, gop2):
        logging.info("Merging duplicate GOPs %d and %d", gop1.id, gop2.id)
        VFS.instance().database.execute(
            'UPDATE gops SET examined=9999999, filename=? WHERE id = ?', (gop1.filename, gop2.id))
        #TODO
        os.remove(gop2.filename)
        gop2.filename = gop1.filename
        return path.getsize(gop1.filename)

    @classmethod
    def _create_image(cls, template=None, height=None, width=None, dtype=np.uint8):
        return np.empty((height if height is not None else template.height,
                         width if width is not None else template.width,
                         PhysicalVideo.CHANNELS), dtype=dtype)
