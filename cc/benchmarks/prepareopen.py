import os
import sys

#mkdir.py n path

n = int(sys.argv[1])

for i in range(n):
  os.mkdir(sys.argv[2] + '/' + str(i))
  open(sys.argv[2] + '/' + str(i) + '/1x1.hevc', 'w')
