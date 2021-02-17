import os
import logging
from vfs.engine import VFS
from vfs.videoio import get_shape, get_shape_and_codec, split_video, join_video, extensions, encoded


class PhysicalVideo(object):
    CHANNELS = 3

    _physical_video_count = None
    @classmethod
    def count(cls):
        if cls._physical_video_count is None:
            cls._physical_video_count = VFS.instance().database.execute(
                "SELECT COUNT(*) FROM physical_videos").fetchone()[0] or 0
        return cls._physical_video_count

    @classmethod
    def add(cls, logical_video, height, width, codec):
        cls._physical_video_count = cls.count() + 1
        return PhysicalVideo(VFS.instance().database.execute(
            'INSERT INTO physical_videos(logical_id, height, width, codec) '
            'VALUES (?, ?, ?, ?)',
            (logical_video.id, height, width, codec)).lastrowid, logical_video.id, height, width, codec)

    @classmethod
    def _gop_filename_template(cls, logical, physical, gop_id):
        container = '.mp4' if encoded[physical.codec] else ''
        return '{}-{}-{}.{}{}'.format(logical.name, physical.id, gop_id, extensions[physical.codec], container)

    @classmethod
    def load(cls, logical_video, filename, resolution=None, codec=None, fps=None):
        from vfs.gop import Gop

        if resolution is None and codec is None:
            resolution, codec, fps = get_shape_and_codec(filename)
        elif codec is None:
            resolution = get_shape(filename)

        assert(resolution is not None)
        assert(codec is not None)
        assert(fps is not None)

        physical = cls.add(logical_video, *resolution, codec=codec)

        output_filename_template = os.path.join(
            VFS.instance().path,
            cls._gop_filename_template(logical_video, physical, '%04d'))
            #'{}-{}-%04d.{}'.format(logical_video.name, physical.id, extensions[physical.codec]))
        gop_filenames = split_video(filename, output_filename_template, resolution, codec, fps)

        Gop.addmany(physical, [(filename, start_time, end_time, os.path.getsize(filename), fps, 0, 0, None)
                               for (filename, start_time, end_time) in gop_filenames])

        logging.info('Ingested physical video %s-%d', logical_video.name, physical.id)

        return physical

    #def write_to(self, filename):
    #    gops = list(self.gops())
    #
    #    if not any(gop.joint for gop in gops):
    #        join_video((gop.filename for gop in gops), filename)
    #    else:
    #        with tempfile.TemporaryDirectory() as temp_path:
    #            for joint_gop in (gop for gop in gops if gop.joint):
    #                temp_filename = os.path.join(
    #                    temp_path, os.path.basename(joint_gop.filename.format('original')))
    #                temp_filename = 'foo.mp4' #TODO
    #                VFS.instance().compression.co_decompress(joint_gop, temp_filename)
    #                joint_gop.filename = temp_filename
    #
    #            join_video(([gop.filename for gop in gops]), filename)

    @classmethod
    def get(cls, id):
        return PhysicalVideo(*VFS.instance().database.execute(
            'SELECT id, logical_id, height, width, codec '
            'FROM physical_videos WHERE id = ?', id).fetchone())

    @classmethod
    def get_all(cls):
        return map(lambda args: PhysicalVideo(*args),
                   VFS.instance().database.execute(
                       'SELECT id, logical_id, height, width, codec '
                       'FROM physical_videos').fetchall())

    @classmethod
    def delete(cls, video):
        from vfs.gop import Gop

        logging.info('Deleting Physical Video %d', video.id)

        gops = video.gops()
        references = {filename: count for (filename, count) in
                      VFS.instance().database.execute(
                          'SELECT filename, COUNT(*) '
                                 'FROM gops '
                                 'WHERE filename IN (SELECT filename FROM gops WHERE physical_id = ?) '
                                 'GROUP BY filename', video.id).fetchall()}

        VFS.instance().database.execute('DELETE FROM gops WHERE physical_id = ?', video.id)
        VFS.instance().database.execute('DELETE FROM physical_videos WHERE id = ?', video.id)
        video.id = None

        for gop in gops:
            Gop.delete(gop, references=references.get(gop.filename, 0))

    def __init__(self, id, logical_id, height, width, codec):
        self.id = id
        self.logical_id = logical_id
        self.height = height
        self.width = width
        self.codec = codec
        self._logical = None
        self._start_time = None
        self._end_time = None
        self._gops = None

    def logical(self):
        from vfs.logicalvideo import LogicalVideo
        if self._logical is None:
            self._logical = LogicalVideo.get(self.logical_id)
        return self._logical

    def gops(self):
        from vfs.gop import Gop

        if self._gops is None:
            self._gops = list(map(lambda args: Gop(*args),
                       VFS.instance().database.execute(
                           'SELECT id, physical_id, filename, start_time, end_time, cluster_id, joint, examined, '
                           '       histogram, descriptors, keypoints, size, zstandard, fps, mse, estimated_mse, parent_id, original_size '
                           'FROM gops WHERE physical_id = ? ORDER BY id', self.id).fetchall()))
        return self._gops

    def start_time(self):
        if self._start_time is None:
            self._start_time, self._end_time = VFS.instance().database.execute(
                'SELECT MIN(start_time), MAX(end_time) FROM gops WHERE physical_id = ?', self.id).fetchone()
        return self._start_time

    def end_time(self):
        if self._end_time is None:
            self._start_time, self._end_time = VFS.instance().database.execute(
                'SELECT MIN(start_time), MAX(end_time) FROM gops WHERE physical_id = ?', self.id).fetchone()
        return self._end_time

    def resolution(self):
        return self.height, self.width

    def shape(self):
        return self.height, self.width, self.CHANNELS

    def size(self):
        return VFS.instance().database.execute('SELECT SUM(size) FROM gops WHERE physical_id = ?', self.id).fetchone()[0]