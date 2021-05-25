import vfs
from vfs import engine, interop
from vfs.mp4 import MP4
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    with engine.VFS(transient=True):
        vfs.write('v', 'v3.mp4')
        vfs.read('v', 'out.rgb', codec='rgb', fps=30) # t=(0.5, 1), resolution=(320, 320))
        array = vfs.load('v')
        print(array)
        array = array[5:]
        print(array)
        array = array[:10]
        print(array)
        array = array.at('4K')
        print(array)
        array = array[:, 100:200, :]
        print(array)
        array = array.at('2K')
        print(array)

    exit(0)

    with MP4('v3.mp4', required_atoms=[b'mdat', b'stsz', b'avcC', b'avc1']) as mp4:
        #out = interop.open('out2.h264')
        #interop.write(mp4.file.fileno()) #, open('out.h264', 'wb'), 30, 60, mp4.mdat.start, mp4.stsz.offsets)
        #media = mp4.media(30, 60)
        print(mp4.avc1.metadata.width)
        exit(0)

    offsets = interop.unmux('v3.mp4')
    print(offsets)
    with open('v2.mp4', 'rb') as f, open('out.h264', 'wb') as out:
        start = offsets[0]
        end = offsets[10]
        print(start, end)
        f.seek(start)
        data = f.read(end - start)
        out.write(data)

    exit(0)

    with engine.VFS(transient=True):
        array = vfs.load('v')
        print(array)
        array = array[5:]
        print(array)
        array = array[:10]
        print(array)
        array = array.at('4K')
        print(array)
        array = array[:, 100:200, :]
        print(array)
        array = array.at('2K')
        print(array)
        #array = array[frame()]

        #a = array.to_numpy('h264')

        #print(a.shape)

        a = list(array.array_split(3))
        print(len(list(a)))
        print('x')

        #for frame in array:
        #    print(frame.shape)

