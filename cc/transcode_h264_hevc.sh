#!/bin/bash

rm list.txt
for f in ${1}/*.h264; do echo "file '$f'" >> list.txt; done

ffmpeg -hide_banner -loglevel panic -f concat -safe 0 -i ./list.txt -codec nvenc_hevc -f rawvideo -

