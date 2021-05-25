from .context import vfs, TEST_VIDEO_MP4

import os
import unittest
import logging


class MP4TestSuite(unittest.TestCase):
    logging.basicConfig(level=logging.DEBUG)

    def test_demux(self):
        with vfs.mp4.MP4(TEST_VIDEO_MP4, required_atoms=[b'mdat', b'stsz', b'avcC', b'avc1']) as mp4:
            assert(mp4.avc1.metadata.width == 960)
            assert(mp4.avc1.metadata.height == 540)
            assert(mp4.codec == 'h264')
            assert(mp4.mdat.size > 0)
            assert(len(mp4.stsz.offsets) > 0)


if __name__ == '__main__':
    unittest.main()