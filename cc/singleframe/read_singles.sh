#!/bin/bash

ffmpeg -f concat -safe 0 -pix_fmt rgb24 -s 960x540 -i ./list.txt -f rawvideo -f null -

