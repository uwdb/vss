import time
import logging
import cv2

def roundeven(n):
    n = int(round(n))
    return n + n % 2

def psnr(left, right):
    return cv2.PSNR(left, right)

def log_runtime(text, level=logging.INFO):
    class LogRuntime:
        def __init__(self, label, level):
            self.label = label
            self.level = level
            self.duration = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, *args):
            self.duration = time.time() - self.start_time
            logging.log(self.level, f'{self.label} completed in {self.duration:0.2f}')

    return LogRuntime(text, level)