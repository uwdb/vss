import sys

size = int(sys.argv[3]) * int(sys.argv[4]) * 3
i = 0

while True:
  with open(sys.argv[1], "rb") as f:
    with open(sys.argv[2] % i, "wb") as o:
        chunk = f.read(size)
        if len(chunk) < size:
            exit(1)
        o.write(chunk)
        print(i)
        i += 1
