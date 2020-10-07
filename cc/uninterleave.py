import sys

infn = sys.argv[1]
leftfn = sys.argv[2]
rightfn = sys.argv[3]
frame_size = int(sys.argv[4]) * int(sys.argv[5]) * 3
i = 0

with open(leftfn, "wb") as lf:
    with open(rightfn, "wb") as rf:
        with open(infn, "rb") as inf:
            while True:
                f1 = inf.read(frame_size)
                f2 = inf.read(frame_size)
                lf.write(f1)
                rf.write(f2)
                print(i, len(f1), len(f2))
                i += 1

                if len(f1) == 0 or len(f2) == 0:
                    break

                f1 = f2 = None
