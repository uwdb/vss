import os
import sys
import random

# open.py n m filename path

n = int(sys.argv[1])
m = int(sys.argv[2])
filename = sys.argv[3]
path = sys.argv[4]

for i in range(n):
  r = random.randint(0, m - 1)
  open(path + '/' + str(r) + '/' + filename)
