# VSS: A Storage System for Video Analytics

VSS is a new video storage system designed to decouple
high-level video operations from the low-level details required to
store and efficiently retrieve video data. VSS is designed to be the
storage subsystem of a video data management system
and is responsible for: 

1. Transparently and automatically arranging
the data on disk in an efficient, granular format
2. Caching
frequently-retrieved regions in the most useful formats
3. Eliminating redundancies found in videos captured from multiple
cameras with overlapping fields of view. 

Our results suggest that
VSS can *improve VDBMS read performance by up to 54%*, *reduce
storage costs by up to 45%*, and enable developers to *focus on
application logic* rather than video storage and retrieval.

Please see the [project website](https://db.cs.washington.edu/projects/visualworld) for more details about VSS, links to the papers, and related projects.

## Installation

VSS is distributed as a Python module, with the usual installation process:

```bash
git clone https://github.com/uwdb/vss.git
cd vss
python setup.py install # or develop

```

## Example usage

Developers can interact with VSS through both a command line and Python interface.  Both interfaces contain equivalent functionality.

### Python API

VSS exposes a Python API for developers who wish to leverage it in their systems.  The high level API exposes video data as lazily-executed sequence of slices over a Numpy array interface, while the low level API exposes the simple read/write interface described in the [paper](https://db.cs.washington.edu/projects/visualworld/vss.pdf).


In either case, the first step is to create an instance of the VSS engine, e.g.:

```python
import vss

with vss.engine.VSS(transient=True):
    # Your operations here, e.g.:
    vss.write('my_video', 'tests/test_video.mp4')
```

An optional `transient` flag controls the application of auxiliary VSS optimizations such as quality-based eviction or deferred compression.  More fine-grained options are also available; see [engine.py](https://github.com/uwdb/vss/blob/master/vfs/engine.py).

#### High level API

The `load` operation is used to retrieve video data from VSS as a Numpy-like array.  Arrays are dynamic structures with exactly three dimensions, respectively duration in seconds, vertical resolution, horizontal resolution.  For example, if `v = vss.load(...)` produces a ten-second 4K video, `v.shape = (10, 2160, 3840)`.

Each 3D VSS array supports spatial (i.e., to select a region of interest) or temporal (i.e., to select a time range) slicing.  Slice steps also enable downsampling (by stepping over the spatial dimensions) and framerate adjustments (by stepping over the temporal dimension).

Operations such as slicing are lazily applied to VSS arrays.
Once the desired data has been selected, a developer converts a VSS array into a Numpy array using the `to_array()` method.  This method invokes the lower-level API (described below) to retrieve the requested video data and transforms it into a Numpy array.

Some simple examples:

```python
with vss.engine.VSS(transient=True):
    vss.write('v', 'tests/test_video.mp4')

    array = vss.load('v')

    array = array[5:]            # Select the last five seconds of the video
    array = array[:1]            # Select the first second of the intermediate result, i.e. time=[5, 6)
    array = array.at('2K')       # Subsample at 2K resolution; equivalent to array[:, ::2, ::2]
    array = array[:, 0:50, 0:50] # Select the top-left 50x50 region of the downsampled video
    array = array[:, :, ::30]    # Sample at 30FPS

    numpy_array = array.to_array() # Materialize as a 3x50x50x1 Numpy array (3-channel RGB, t, y, x)
```

More examples are available in the [VSS array tests](https://github.com/BrandonHaynes/vss/tree/master/tests/test_array.py).

#### Low level API

The VSS low level API exposes simple read/write operations that are identical to API described in the research paper.  The following example illustrates writing a video into VSS and then reading the first five seconds at 1K resolution:

```python
with vss.engine.VSS(transient=True):
    vss.write('v', 'tests/test_video.mp4')
    data = vss.read('v', resolution=(540, 960), t=(0, 5))
``` 

VSS also supports listing and deleting videos stored in its internal catalog:

```python
vss.list() # ['v']
vss.delete('v')
``` 

### Command line API

VSS also exposes command line API for basic read/write operations.  Read operations may optionally contain spatial (e.g., `--resolution`, `--roi`, `--times`) or physical (e.g., codec) constraints.  Execute `vss read --help` for more details.

#### Write a video into VSS

```bash
vss write [name] [local filesystem filename]
```

#### Read a video from VSS

```bash
vss read [name] [constraints] [output filesystem filename]
```

#### List videos stored in VSS

```bash
vss list
```

#### Delete a video in VSS

```bash
vss delete [name]
```

Execute `vss --help` for details about additional parameters and query constraints available in the command line API.

## Citations & Paper

If you use VSS, please cite our SIGMOD'21 paper:

_VSS: A Storage System for Video Analytics_<br />
Brandon Haynes, Maureen Daum, Dong He, Amrita Mazumdar, Magdalena Balazinska, Alvin Cheung, Luis Ceze<br />
SIGMOD:685-696 [[PDF]](https://doi.org/10.1145/3448016.3459242)

```
@inproceedings{DBLP:conf/sigmod/HaynesDHMBCC21,
  author    = {Brandon Haynes and
               Maureen Daum and
               Dong He and
               Amrita Mazumdar and
               Magdalena Balazinska and
               Alvin Cheung and
               Luis Ceze},
  title     = {{VSS:} {A} Storage System for Video Analytics},
  booktitle = {{SIGMOD}},
  pages     = {685--696},
  year      = {2021},
  doi       = {10.1145/3448016.3459242},
}
```

See the [project website](https://db.cs.washington.edu/projects/visualworld) for more details about VSS, links to the papers, and related projects.
