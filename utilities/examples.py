import vfs
from vfs import engine

if __name__ == '__main__':
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

