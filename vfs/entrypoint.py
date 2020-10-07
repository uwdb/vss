#!/usr/bin/env python3

import time
import re
import argparse
import logging
from vfs.api import read, write, list, delete
from vfs.videoio import extensions, H264, HEVC, RGB8
from vfs.engine import VFS
from vfs.db import Database

def regex_type(pattern):
    pattern = re.compile(pattern)
    def evaluate(value):
        if not pattern.match(value):
            raise argparse.ArgumentTypeError
        return value
    return evaluate

def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'mode',
        choices=['init', 'read', 'write', 'list', 'delete', 'start', 'test'],
        help='Video operation mode')
    parser.add_argument(
        'name',
        nargs='?',
        help='Video name to read, write, or delete')
    parser.add_argument(
        'filename',
        nargs='?',
        help='Video name to read or write')
    parser.add_argument(
        '-e', '--codec', type=regex_type('|'.join(extensions.values())),
        help='Video resolution (read only)')
    parser.add_argument(
        '-r', '--resolution', type=regex_type('\d+x\d+'),
        help='Video resolution (read only)')
    parser.add_argument(
        '-t', '--time', type=regex_type('\d+?-\d+?'),
        help='Video times (read only)')
    parser.add_argument(
        '-c', '--crop', type=regex_type('\d+:\d+:\d+:\d+'),
        help='Video crop (read only)')
    parser.add_argument(
        '-fps', '--fps', type=int,
        help='Video FPS (read only)')
    arguments = parser.parse_args() #['read', 'p3', 'foo.mp4'])

    if arguments.mode == 'init':
        Database.clear()
    else:
        with VFS() as instance:
            time.sleep(1)

            if arguments.mode == 'start':
                while True:
                    time.sleep(60)
            elif arguments.mode == 'list':
                list()
            elif arguments.mode == 'read':
                read(arguments.name, arguments.filename,
                     resolution=tuple(map(int, arguments.resolution.split('x'))) if arguments.resolution else None,
                     roi=tuple(map(int, arguments.crop.split(':'))) if arguments.crop else None,
                     t=tuple(map(int, arguments.time.split('-'))) if arguments.time else None,
                     codec=arguments.codec,
                     fps=arguments.fps)
            elif arguments.mode == 'write':
                write(arguments.name, arguments.filename)
            elif arguments.mode == 'delete':
                delete(arguments.name)
            elif arguments.mode == 'test':
                #write('p2', "inputs/p2.rgb", (2160, 3840), RGB8, 30)
                write('p1', "inputs/p1.mp4")
                write('p2', "inputs/p2.mp4")

                read('p2', 'output.mp4', resolution=(1000, 1000), roi=(20,20,500,500), t=(0, 1), codec=H264)
                read('p2', 'output2.mp4', resolution=(1000, 1000), roi=(20,20,500,500), t=(1, 3), codec=H264, fps=10)
        time.sleep(1)

if __name__ == '__main__':
    main()