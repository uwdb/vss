import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
TEST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests')
TEST_VIDEO_MP4 = os.path.join(TEST_PATH, 'test_video.mp4')

import vfs