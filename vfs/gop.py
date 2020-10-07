import os
import logging
from vfs.engine import VFS
from vfs.physicalvideo import PhysicalVideo
from vfs.jointcompression import JointCompression

class Gop(object):
    @classmethod
    def add(cls, physical_video, filename, start_time, end_time, size, fps):
        gop = Gop(None, physical_video.id, filename, start_time, end_time, None, False, False, None, None, None, size, None, fps, None, None, None)
        gop.id = VFS.instance().database.execute(
            'INSERT INTO gops(physical_id, filename, start_time, end_time, size, fps) '
                            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                            (gop.physical_id, gop.filename, gop.start_time, gop.end_time, gop.size, gop.fps, gop.mse, gop.estimated_mse, gop.parent_id)).lastrowid
        return gop

    @classmethod
    def addmany(cls, physical_video, data):
        #TODO SQL injection :(
        VFS.instance().database.executebatch(
            ('INSERT INTO gops(physical_id, filename, start_time, end_time, size, fps, mse, estimated_mse, parent_id) ' 
                            "VALUES ({}, '{}', {}, {}, {}, {}, {}, {}, {})".format(
                physical_video.id, filename, start_time, end_time, size, fps,
                mse if mse is not None else 'NULL', estimated_mse if estimated_mse is not None else 'NULL',
                parent_id if parent_id is not None else 'NULL')
                      for filename, start_time, end_time, size, fps, mse, estimated_mse, parent_id in data))

    @classmethod
    def get(cls, id):
        return Gop(*VFS.instance().database.execute(
            'SELECT id, physical_id, filename, start_time, end_time, cluster_id, joint, examined, '
                               '       histogram, keypoints, descriptors, size, zstandard, fps, mse, estimated_mse, parent_id '
                               'FROM gops WHERE id = ?', id).fetchone())

    @classmethod
    def delete(cls, gop, references=None):
        references = (references or VFS.instance().database.execute(
            'SELECT COUNT(*) FROM gops WHERE filename = ?', gop.filename).fetchone()[0])
        logging.info('Deleting GOP %d (underlying data has %d references)', gop.id, references)

        if gop.id:
            VFS.instance().database.execute('DELETE FROM gops WHERE id = ?', gop.id)
            gop.id = None

        if not gop.joint:
            #logging.warning("Skipping deleting physical GOP file in debug mode")
            os.remove(gop.filename)
            #pass
        elif references <= 1:
            logging.info("Last joint compressed candidate deleted; removing physical file")
            #logging.warning("Skipping deleting physical GOP file in debug mode")
            os.remove(gop.filename.format(JointCompression.LEFT))
            #os.remove(gop.filename.format(JointCompression.RIGHT))
            #os.remove(gop.filename.format(JointCompression.OVERLAP))
        else:
            logging.info("Not removing physical file; other references exist")

    def __init__(self, id, physical_id, filename, start_time, end_time, cluster_id, joint, examined, histogram, keypoints, descriptors,
                 size, zstandard, fps, mse, estimated_mse, parent_id):
        self.id = id
        self.physical_id = physical_id
        self.filename = filename
        self.start_time = start_time
        self.end_time = end_time
        self.cluster_id = cluster_id
        self.examined = examined
        self.joint = joint
        self.histogram = histogram
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.size = size
        self.zstandard = zstandard
        self.fps = fps
        self.mse = mse
        self.estimated_mse = estimated_mse
        self.parent_id = parent_id
        self._video = None

    def video(self):
        if self._video is None:
           self._video = PhysicalVideo.get(self.physical_id)
        return self._video