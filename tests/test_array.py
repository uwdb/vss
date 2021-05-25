from .context import vfs, TEST_VIDEO_MP4

import logging
import unittest

class ArrayTestSuite(unittest.TestCase):
    logging.basicConfig(level=logging.DEBUG)

    def test_shape(self):
        vfs.db.Database.clear()

        with vfs.engine.VFS(transient=True):
            vfs.write('v', TEST_VIDEO_MP4)

            array = vfs.load('v')
            assert(array.shape[0] == 6)
            assert(array.shape[1:] == (540, 960))

    def test_time(self):
        vfs.db.Database.clear()

        with vfs.engine.VFS(transient=True):
            vfs.write('v', TEST_VIDEO_MP4)

            array = vfs.load('v')

            array = array[5:]  # t = [5, 6)
            assert(array.shape[0] == 1)
            array = array[:10] # t = [5, 6)
            assert(array.shape[0] == 1)

    def test_resolution(self):
        vfs.db.Database.clear()

        with vfs.engine.VFS(transient=True):
            vfs.write('v', TEST_VIDEO_MP4)

            array = vfs.load('v')

            assert(array.shape[1:] == vfs.array.resolutions['1K'])
            assert(array.at('2K').shape[1:] == vfs.array.resolutions['2K'])
            assert(array.at('4K').shape[1:] == vfs.array.resolutions['4K'])

    def test_crop(self):
        vfs.db.Database.clear()

        with vfs.engine.VFS(transient=True):
            vfs.write('v', TEST_VIDEO_MP4)

            array = vfs.load('v')

            array = array[:, 100:200, :]   # 100 x 960
            assert (array.shape[1:] == (100, 960))
            array = array[:, 0:50, :]   # 50 x 960
            assert (array.shape[1:] == (50, 960))

            array = array[:, :, 100:200]   # 50 x 100
            assert (array.shape[1:] == (50, 100))
            array = array[:, :, 0:50]   # 50 x 50
            assert (array.shape[1:] == (50, 50))

#    def test_crop(self):
#        vfs.db.Database.clear()

#        with vfs.engine.VFS(transient=True):
 #           vfs.write('v', TEST_VIDEO_MP4)

  #          array = vfs.load('v')

            # array = array[frame()]


if __name__ == '__main__':
    unittest.main()