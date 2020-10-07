#!/bin/bash

rm list.txt
for f in ${1}/*.h264; do echo "file '$f'" >> list.txt; done

ffmpeg -hide_banner -loglevel panic -f concat -safe 0 -i ./list.txt -pix_fmt rgb24 -f rawvideo -
#ffmpeg -hide_banner -loglevel panic -f concat -safe 0 -i ./list.txt -pix_fmt rgb24 -codec nvenc_h264 -f rawvideo -

