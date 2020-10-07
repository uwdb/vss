import sys

n = int(sys.argv[1])

with open('list.txt', "w") as f:
    for i in range(n):
      f.write("file '%d.rgb'\n" % (i + 1))
