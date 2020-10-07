import sys

leftfn = sys.argv[1]
rightfn = sys.argv[2]
outfn = sys.argv[3]
frame_size = int(sys.argv[4]) * int(sys.argv[5]) * 3
i = 0

with open(leftfn, "rb") as lf:
    with open(rightfn, "rb") as rf:
        with open(outfn, "wb") as out:
            while True:
                l = lf.read(frame_size)
                r = rf.read(frame_size)
                print(i, len(l), len(r))
                i += 1

                if len(l) == 0 or len(r) == 0:
                    break

                out.write(l)
                out.write(r)
                l = None
                r = None