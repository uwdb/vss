import time
import sys
import subprocess
import random
import logging
from vfs import api
from vfs import engine

def create_cache(n, name, T, R, P):
    with engine.VFS(transient=True):
        for i in range(n):
            r = random.choice(R)
            p = random.choice(P)
            t1 = random.randint(0, T-1)
            t2 = min(random.randint(t1 + 1, t1 + 60), T - 1)

            with open('cache.txt', 'w') as f:
                f.writelines([f'{i} cache {name} {(t1, t2)} {r} {p}\n'])

            print(f'{i} cache {name} {(t1, t2)} {r} {p}')
            api.read(name, f"out.{p}", resolution=r, t=(t1, t2), codec=p)

def read_vfs(i, source, r, t, p):
    #r = (2160, 3840)
    #t = (1990, 1991)
    #p = 'h264'
    t = (t[0] + 0.7, t[1] + 0.7)
    api.read(source, '/tmp/out.mp4', resolution=r, t=t, codec=p)

def read_naive(i, source, r, t, p):
    start_time = time.time()

    subprocess.check_call(f'ffmpeg -y -ss {t[0]} -to {t[1]} -i {source} -vf scale={r[1]}:{r[0]} /tmp/out.{p}',
                          shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    duration = time.time() - start_time
    print(f'{i} naive duration {duration:02f}')

def read_random(i, reader, source, r, t, p):
    reader(i, source, r, (t, t+1), p)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    R = [(540, 960), (1080, 1920), (2160, 3840)]
    P = ['h264', 'hevc']
    T = (0, 3600)

    with engine.VFS(transient=True):
        api.vacuum()

    #create_cache(500, "v", 3600, R, P)
    #exit(0)

    n = 50
    ts = [random.randint(*T) for _ in range(n)]
    ps = [random.choice(P) for _ in range(n)]
    rs = [random.choice(R) for _ in range(n)]

    apply_deferred_compression = False
    apply_eviction = True

    #with engine.VFS(transient=True):
    #    if 'v' not in api.list():
    #        api.write('v', 'inputs/visualroad-4k-30a.mp4')

    durations = []
    with engine.VFS(transient=False,
                    apply_eviction=apply_eviction,
                    apply_deferred_compression=apply_deferred_compression,
                    create_metadata=False,
                    apply_joint_compression=False,
                    apply_compaction=False,
                    verify_quality=False):
        for i in range(n):
            #durations.append(read_random(i, read_naive, 'inputs/visualroad-4k-30a.mp4', rs[i], ts[i], ps[i]))
            durations.append(read_random(i, read_vfs, 'v', rs[i], ts[i], ps[i]))



    #with engine.VFS(transient=True):
    #    read_vfs('v', R[0], (0, 10), P[0])
