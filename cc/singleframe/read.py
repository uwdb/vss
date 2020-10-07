import sys

n = int(sys.argv[1])

for i in range(n):
    with open('%d.rgb' % ((i + 1) % 2000), "rb") as f:
        f.read()
