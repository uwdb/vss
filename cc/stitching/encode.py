import sys
import subprocess
import os
import re

dir = sys.argv[1]
files = [f for f in os.listdir(dir) if '.png' in f]
categories = list(set([re.sub("\d+\.png", '', f) for f in files]))
categories = list(set([re.sub("\.h265", '', c) for c in categories]))
codec = 'libx265' #'nvenc_hevc'

print codec

for category in categories:
    print category
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-i', os.path.join(dir, category + '%03d.png'), '-codec', codec, os.path.join(dir, (category or 'out') + '.h265')]
    print ' '.join(command)

    subprocess.check_call(command)
