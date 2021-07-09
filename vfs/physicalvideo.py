import os
import shutil
import logging
from vfs.engine import VFS
from vfs.videoio import extensions, encoded, frame_size
from vfs.mp4 import MP4

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
    def add(cls, logical_video, height, width, codec, filename, headers, gop_size, mdat_offset):
        cls._physical_video_count = cls.count() + 1
        return PhysicalVideo(VFS.instance().database.execute(
            'INSERT INTO physical_videos(logical_id, height, width, codec, filename, headers, gop_size, mdat_offset) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (logical_video.id, height, width, codec, filename, headers, gop_size, mdat_offset)).lastrowid, logical_video.id, height, width, codec, filename, headers, gop_size, mdat_offset)

    #@classmethod
    #def _gop_filename_template(cls, logical, physical, gop_id):
    #    container = '.mp4' if encoded[physical.codec] else ''
    #    return '{}-{}-{}.{}{}'.format(logical.name, physical.id, gop_id, extensions[physical.codec], container)

    @classmethod
    def _ingest_filename(cls, logical, physical):
        container = '.mp4' if encoded[physical.codec] else ''
        return '{}-{}.{}{}'.format(logical.name, physical.id, extensions[physical.codec], container)

    @classmethod
    def load(cls, logical_video, filename, resolution=None, codec=None, fps=None, copy=True):
        from vfs.gop import Gop

        if codec is None or encoded[codec]:
            with MP4(filename) as mp4:
                resolution = resolution or (mp4.height, mp4.width)
                codec = codec or (mp4.codec)
                fps = fps or mp4.fps
                headers = mp4.headers
                mdat_offset = mp4.mdat.start
                gop_size = mp4.gop_size or mp4.frame_count
                gop_byte_offsets = mp4.stsz.offsets[::gop_size] + [mp4.mdat.size]
                gop_byte_windows = zip(gop_byte_offsets, gop_byte_offsets[1:])
                gop_times = (((index * gop_size) / fps, ((index + 1) * gop_size) / fps) for index in range(len(mp4.stsz.offsets) // gop_size))
        else:
            assert(resolution is not None)
            assert(codec is not None)
            mdat_offset = 0
            headers = b''
            gop_byte_windows = [(0, os.path.getsize(filename))]
            gop_size = gop_byte_windows[0][1] // frame_size(codec, resolution)
            gop_times = [(0, gop_size // fps)]

        assert(resolution is not None)
        assert(codec is not None)
        assert(fps is not None)

        physical = cls.add(logical_video, *resolution, codec=codec, filename=None, headers=headers, gop_size=gop_size, mdat_offset=mdat_offset)
        output_filename = os.path.join(VFS.instance().path, cls._ingest_filename(logical_video, physical))

        #output_filename_template = os.path.join(
        #    VFS.instance().path,
        #    cls._gop_filename_template(logical_video, physical, '%04d'))
        #gop_filenames = split_video(filename, output_filename_template, resolution, codec, fps)
        if copy:
            shutil.copy(filename, output_filename)
        else:
            shutil.move(filename, output_filename)

        VFS.instance().database.execute("UPDATE physical_videos SET filename=? WHERE id = ?", (output_filename, physical.id))
        physical.filename = output_filename


        Gop.addmany(physical, [(filename, start_time, end_time, start_byte_offset, end_byte_offset, os.path.getsize(output_filename), fps, 0, 0, None)
                               for ((start_byte_offset, end_byte_offset), (start_time, end_time)) in zip(gop_byte_windows, gop_times)])

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
            'SELECT id, logical_id, height, width, codec, filename, headers, gop_size, mdat_offset '
            'FROM physical_videos WHERE id = ?', id).fetchone())

    @classmethod
    def get_all(cls):
        return map(lambda args: PhysicalVideo(*args),
                   VFS.instance().database.execute(
                       'SELECT id, logical_id, height, width, codec, filename, headers, gop_size, mdat_offset '
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

    def __init__(self, id, logical_id, height, width, codec, filename, headers, gop_size, mdat_offset):
        self.id = id
        self.logical_id = logical_id
        self.height = height
        self.width = width
        self.codec = codec
        self.filename = filename
        self.headers = headers
        self.mdat_offset = mdat_offset
        self.gop_size = gop_size
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
                           '       histogram, descriptors, keypoints, size, zstandard, fps, mse, estimated_mse, parent_id, original_size, start_byte_offset, end_byte_offset '
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
