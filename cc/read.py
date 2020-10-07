import sys

fn = sys.argv[1]
read = 0
size = 4*1024*1024
i = 0

with open(fn) as f:
    while True:
        #data = f.read(size)
        f.read(size)
        i += 1
        #if len(data) == 0:
        if i == 684:
            break
        #read += len(data)

#print(read)
