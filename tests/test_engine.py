from .context import vfs

import logging
import unittest

class EngineTestSuite(unittest.TestCase):
    logging.basicConfig(level=logging.DEBUG)

    def test_default_engine_init(self):
        vfs.db.Database.clear()
        with vfs.engine.VFS():
            pass

    def test_transient_engine_init(self):
        vfs.db.Database.clear()
        with vfs.engine.VFS(transient=True):
            pass


if __name__ == '__main__':
    unittest.main()