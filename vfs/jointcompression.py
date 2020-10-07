import os
from os import path
import threading
import logging
import numpy as np
import cv2
from vfs.videoio import VideoReader, VideoWriter, NullReader, NullWriter
from vfs.engine import VFS
from vfs.physicalvideo import PhysicalVideo
#from vfs.gop import Gop
from vfs.descriptor import Descriptor
import vfs.homography
from vfs.utilities import roundeven

class JointCompression(object):
    LEFT = 'left'
    RIGHT = 'right'
    OVERLAP = 'overlap'

    @classmethod
    def get_candidate(cls, epoch):
        from vfs.gop import Gop

        smallest_cluster = (VFS.instance().database.execute(
            'SELECT cluster_id, COUNT(*) FROM gops candidate '
              'WHERE examined <= ? AND joint = 0 AND '
              '      cluster_id > 0 AND '
              '      physical_id < (SELECT MAX(physical_id) FROM gops WHERE examined <= ? AND joint = 0 AND cluster_id = candidate.cluster_id) '
              'GROUP BY cluster_id ORDER BY COUNT(*) ASC', (epoch, epoch)).fetchone() or [None])[0]

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

    output = open('joint.csv', "w")

    @classmethod
    def estimate_homography(cls, gop1, gop2, matches=None, frame1=None, frame2=None):
        if matches is None:
            keypoints1, descriptors1 = Descriptor.create(frame1)
            keypoints2, descriptors2 = Descriptor.create(frame2)
            has_homography, matches = Descriptor.adhoc_match(descriptors1, descriptors2)
            keypoints1, keypoints2 = keypoints1, keypoints2
        else:
            keypoints1, keypoints2 = gop1.keypoints, gop2.keypoints
            has_homography = True

        H, Hi = vfs.homography.project(keypoints1, keypoints2, matches)

        frame2_overlap_yoffset = -int(round(np.dot([0, gop1.video().height, 1], Hi)[0]))
        frame1_left_width = roundeven(Hi.dot([0,0,1])[0])
        frame2_right_width = max(roundeven(gop2.video().width - H.dot([gop1.video().width, 0, 1])[0] / H.dot([gop1.video().width, 0, 1])[2]), 0)
        overlap_height = gop2.video().height + 2 * frame2_overlap_yoffset
        overlap_width = gop1.video().width - frame1_left_width

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

        left = cls._create_image(gop1.video(), width=frame1_left_width)
        right = cls._create_image(gop2.video(), width=frame2_right_width)
        overlap = cls._create_image(height=overlap_height, width=overlap_width)
        recovered_frame2 = np.empty(gop2.video().shape(), dtype=np.uint8)

        return H, Hi, left, right, overlap, recovered_frame2, frame2_overlap_yoffset, has_homography

    @classmethod
    def co_compress(cls, gop1, gop2, matches, abort_psnr_threshold=25):
        cls.output.write('%s,%d,%d,%s,%d,%d\n' % (gop1.video().logical().name, gop1.video().id, gop1.id,
                     gop2.video().logical().name, gop2.video().id, gop2.id))
        cls.output.flush()
        logging.info('Joint compress %s-%d-%d and %s-%d-%d (%d matches)',
                     gop1.video().logical().name, gop1.video().id, gop1.id,
                     gop2.video().logical().name, gop2.video().id, gop2.id,
                     len(matches))

        assert(gop1.id != gop2.id)
        assert(gop1.video().codec == gop2.video().codec)
        assert(not gop1.joint)
        assert(not gop2.joint)

        H, Hi, left, right, overlap, recovered_frame2, frame2_overlap_yoffset, _ = cls.estimate_homography(gop1, gop2, matches)
        Hs = np.hstack([[0], H.flatten()])

        #H, Hi = homography.project(gop1.keypoints, gop2.keypoints, matches)

        #frame2_overlap_yoffset = -int(round(np.dot([0, gop1.video().height, 1], Hi)[0]))
        #frame1_left_width = roundeven(Hi.dot([0,0,1])[0])
        #frame2_right_width = roundeven(gop2.video().width - H.dot([gop1.video().width, 0, 1])[0] / H.dot([gop1.video().width, 0, 1])[2])
        #overlap_height = gop2.video().height + 2 * frame2_overlap_yoffset
        #overlap_width = gop1.video().width - frame1_left_width

        # Are videos identical?
        if left.shape[1] == 0 and right.shape[1] == 0:
        #if frame1_left_width == 0 and frame2_right_width == 0:
            return cls.deduplicate(gop1, gop2)
        elif left.shape[1] < 0:
            return cls.co_compress(gop2, gop1, matches)
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
            codec = gop1.video().codec

            # if frame1_left_width else NullWriter() as leftwriter, \
            #if frame2_right_width else NullWriter() as rightwriter, \
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
                            np.copyto(overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height],
                                      frame1[:, left.shape[1]:])
                            # Mean join (has a bug in non-overlapping left)
                            #overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height] //= 2
                            #overlap[frame2_overlap_yoffset:frame2_overlap_yoffset + gop1.video().height] += frame1[:, left.shape[1]:] // 2
                            #TODO support interlace mode

                            if left.shape[1] != 0:
                                np.copyto(left, frame1[:, :left.shape[1]])
                            if right.shape[1] != 0:
                                np.copyto(right, frame2[:, -right.shape[1]:])

                            cv2.warpPerspective(overlap, H, dsize=tuple(reversed(frame2.shape[:2])), dst=recovered_frame2)
                            np.copyto(recovered_frame2[:, -right.shape[1]:], frame2[:, -right.shape[1]:])
                            psnr = cv2.PSNR(frame2, recovered_frame2)

                            if psnr < abort_psnr_threshold:
                                if attempts == 2:
                                    abort = True
                                    break
                                else:
                                    logging.info("Recomputing homography (%d, %d)", gop1.id, gop2.id)
                                    #TODO save new homography
                                    H, Hi, left, right, overlap, recovered_frame2, frame2_overlap_yoffset, has_homography = cls.estimate_homography(
                                        gop1, gop2, matches)
                                    np.vstack([Hs, np.hstack([[frame_index], H.flatten()])])
                                    #Hs.append(np.hstack([[frame_index], H.flatten()]))
                            else:
                                break

                        leftwriter.write(left)
                        rightwriter.write(right)
                        overlapwriter.write(overlap)

            if abort:
                logging.info('Joint compression aborted (%d, %d)', gop1.id, gop2.id)
                os.remove(filenametemplate.format(cls.LEFT))
                os.remove(filenametemplate.format(cls.OVERLAP))
                os.remove(filenametemplate.format(cls.RIGHT))
                VFS.instance().database.execute(
                    'UPDATE gops SET examined=9999999 '
                           'WHERE id in (?, ?)',
                           (gop1.id, gop2.id)).close()
                return 0
            else:
                VFS.instance().database.execute(
                    'UPDATE gops SET examined=9999999, joint=1, filename=?, homography=?, shapes=?, is_left=(id=?) '
                           'WHERE id in (?, ?)',
                           (filenametemplate, np.vstack(Hs), np.vstack([left.shape, overlap.shape, right.shape]),
                            gop1.id, gop1.id, gop2.id)).close()

                bytes_saved = (path.getsize(gop1.filename) +
                             path.getsize(gop2.filename) -
                             (path.getsize(filenametemplate.format(cls.LEFT)) +
                              path.getsize(filenametemplate.format(cls.OVERLAP)) +
                              path.getsize(filenametemplate.format(cls.RIGHT))))
                os.remove(gop1.filename)
                os.remove(gop2.filename)
                logging.info('Joint compression saved %dKB', bytes_saved // 1000)
                return bytes_saved

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
                left, overlap, right = leftreader.read(), overlapreader.read(), rightreader.read()

                if overlap is not None:
                    cv2.warpPerspective(overlap, H, dsize=tuple(reversed(overlap.shape[:2])), dst=frame)

                if is_left and left is not None:
                    np.copyto(frame, left)
                elif right is not None:
                    np.copyto(frame[:, -right.shape[1]:], right)

                writer.write(frame)
        pass

    @classmethod
    def deduplicate(cls, gop1, gop2):
        logging.info("Merging duplicate GOPs %d and %d", gop1.id, gop2.id)
        #TODO os.remove(gop2.filename)
        gop2.filename = gop1.filename
        VFS.instance().database.execute(
            'UPDATE gops SET examined=9999999, filename=? WHERE id = ?', (gop1.filename, gop2.id))
        return path.getsize(gop1.filename)

    @classmethod
    def _create_image(cls, template=None, height=None, width=None):
        return np.empty((height if height is not None else template.height,
                         width if width is not None else template.width,
                         PhysicalVideo.CHANNELS), dtype=np.uint8)
