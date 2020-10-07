## Setup: 

```bash
python setup.py develop
```

Note that using VFS requires Python >=3.5 and a Ffmpeg build with Nvidia NVENC en/decoders enabled.  If you get an `invalid decoder` error, you may need to upgrade/rebuild Ffmpeg.

## Commands:

Write video into VFS:
```bash
vfs write [name] [local filesystem filename]
```

Read video from VFS:
```bash
vfs read [name] [local filesystem filename]
```

List contents of VFS:
```bash
vfs list
```

Delete video:
```bash
vfs delete
```

More commands (predicates, filters, engine daemon, etc):
```bash
vfs --help
```

# Python API

```python
import vfs

vfs.list()
```
