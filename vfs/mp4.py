import os
from itertools import accumulate
import struct

from vfs.bitstream import BitStream


MP4_EXTENSION = '.mp4'
UINT32_SIZE = 4
HEADER_SIZE = 8
DEFAULT_ATOMS = {b'mdat', b'stsz', b'avcC', b'avc1'}
NAL_PREFIX = b'\x00\x00\x01'


def write_video_data(input_file, output_file, start_offset, end_offset, mdat_start):
    #start_offset = stsz_offsets[start_frame]
    #bytes_remaining = stsz_offsets[end_frame] - start_offset
    bytes_remaining = end_offset - start_offset

    input_file.seek(mdat_start + start_offset)

    while bytes_remaining > 0:
        frame_data_size = int.from_bytes(input_file.read(UINT32_SIZE), 'big', signed=False)
        frame_data = input_file.read(frame_data_size)

        output_file.write(NAL_PREFIX)
        output_file.write(frame_data)

        bytes_remaining -= frame_data_size + UINT32_SIZE

    assert (bytes_remaining == 0)


class MP4(object):
    def __init__(self, filename, recurse=True, required_atoms=None):
        self.filename = filename
        self.recurse = recurse
        self.atoms = None
        self.required_atoms = set(required_atoms or DEFAULT_ATOMS)
        self._codec = None
        self._mdat = None
        self._stsz = None
        self._avcc = None
        self._avc1 = None
        self._stss = None
        self._height = None
        self._width = None
        self._gop_size = None
        self._frame_count = None
        self._nal_prefix = None
        self._headers = None

    def __enter__(self):
        self.file = open(self.filename, 'rb')
        self.atoms = []

        while self.required_atoms is None or self.required_atoms:
            atom = MP4Atom.parse(self.file, recurse=self.recurse, required_atoms=self.required_atoms)
            if atom is None:
                break
            self.atoms.append(atom)

        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def open(self):
        self.__enter__()

    def close(self):
        self.__exit__(None, None, None)

    @property
    def codec(self):
        if self._codec is None:
            if self.avc1 is not None:
                self._codec = 'h264'
            else:
                self._codec = 'hevc'
        return self._codec

    @property
    def height(self):
        if self._height is None:
            self._height, self._width = self.avc1.metadata.height, self.avc1.metadata.width
        return self._height

    @property
    def width(self):
        if self._width is None:
            self._height, self._width = self.avc1.metadata.height, self.avc1.metadata.width
        return self._width

    @property
    def fps(self):
        return 30

    @property
    def gop_size(self):
        if self._gop_size is None:
            self._gop_size = self.stss.gop_size if self.stss else None
        return self._gop_size

    @property
    def stss(self):
        if self._stss is None:
            self._stss = self.find(MP4SyncSampleAtom)
        return self._stss

    @property
    def mdat(self):
        if self._mdat is None:
            self._mdat = self.find(MP4MovieDataAtom)
        return self._mdat

    @property
    def stsz(self):
        if self._stsz is None:
            self._stsz = self.find(MP4SampleSizeAtom)
        return self._stsz

    @property
    def avcc(self):
        if self._avcc is None:
            self._avcc = self.find(MP4AVCCBox)
        return self._avcc

    @property
    def avc1(self):
        if self._avc1 is None:
            self._avc1 = self.find(MP4AVC1Box)
        return self._avc1

    @property
    def frame_count(self):
        if self._frame_count is None:
            self._frame_count = len(self.stsz.offsets)
        return self._frame_count

    def get_video(self, start_frame, end_frame, include_headers=True):
        start_offset = self.stsz.offsets[start_frame]
        bytes_remaining = self.stsz.offsets[end_frame] - start_offset

        self.file.seek(self.mdat.start + start_offset)

        if include_headers:
            yield self.headers

        while bytes_remaining > 0:
            frame_data_size = int.from_bytes(self.file.read(UINT32_SIZE), 'big', signed=False)
            frame_data = self.file.read(frame_data_size)

            yield self.nal_prefix + frame_data

            bytes_remaining -= frame_data_size + UINT32_SIZE

        assert(bytes_remaining == 0)

    @property
    def nal_prefix(self):
        if self._nal_prefix is None:
            self._nal_prefix = bytes(bytearray(b'\x00' * self.avcc.metadata.nal_length) + b'\x01')
        return self._nal_prefix

    @property
    def headers(self):
        if self._headers is None:
            assert len(self.avcc.sps) == 1
            assert len(self.avcc.pps) == 1
            return self.nal_prefix + self.avcc.sps[0] + self.nal_prefix + self.avcc.pps[0]

        return self._headers

    def findall(self, type):
        for atom in self.atoms:
            yield from atom.findall(type)

    def find(self, type):
        return next(self.findall(type), None)


# TODO don't recurse when we only care about mdat
class MP4Atom(object):
    @classmethod
    def parse(cls, file, recurse=True, required_atoms=None):
        size = int.from_bytes(file.read(4), 'big', signed=False)

        if size == 0:
            return None

        type = file.read(4)
        return types.get(type, MP4Atom)(file, type, size - HEADER_SIZE, recurse=recurse, required_atoms=required_atoms or set())

    def __init__(self, file, type, size, recurse=True, materialize=False, seek=True, required_atoms=None):
        self.start = file.tell()
        self.size = size
        self.type = type

        if materialize:
            self.data = file.read(self.size)
        elif seek:
            self.data = None
            file.seek(self.size, os.SEEK_CUR)
        else:
            self.data = None

        if type in required_atoms:
            required_atoms.remove(type)

    @property
    def full_size(self):
        return self.size + HEADER_SIZE

    @property
    def end(self):
        return self.start + self.size

    def findall(self, type):
        if isinstance(type, bytes) or isinstance(type, str):
            yield from self.findall(types[bytes(type)])
        elif isinstance(self, type):
            yield self
        elif isinstance(self, MP4CompositeAtom) or 'children' in self.__dict__:
            for child in self.children:
                yield from child.findall(type)

    def find(self, type):
        return next(self.findall(type), None)


class MP4FileTypeAtom(MP4Atom):
    pass

class MP4CompositeAtom(MP4Atom):
    #Metadata = namedtuple('ftyp', ('major', 'version', 'brands'))

    def __init__(self, file, type, size, metadata=None, materialize=False, recurse=True, required_atoms=None):
        super(MP4CompositeAtom, self).__init__(file, type, size, seek=False, materialize=False, recurse=recurse, required_atoms=required_atoms)

        assert(materialize == False)
        self.children = self.parse_children(file, self.size, recurse, required_atoms) if recurse else None

    @classmethod
    def parse_children(cls, file, parse_size, recurse_children, required_atoms=None):
        children, remaining = [], parse_size
        while remaining > 0 and (required_atoms is None or  required_atoms):
            atom = MP4Atom.parse(file, recurse=recurse_children, required_atoms=required_atoms)
            assert(atom is not None)
            children.append(atom)
            remaining -= atom.full_size

        assert(remaining == 0 or (required_atoms or True))
        return children


class MP4MovieAtom(MP4CompositeAtom):
    pass


class MP4MovieFormatAtom(MP4CompositeAtom):
    pass


class MP4MovieDataAtom(MP4Atom):
    pass


class MP4MovieFragmentRandomAccessAtom(MP4CompositeAtom):
    pass


class MP4MoveHeaderAtom(MP4Atom):
    pass


class MP4TrackAtom(MP4CompositeAtom):
    pass


class MP4TrackHeaderAtom(MP4Atom):
    pass


class MP4EditAtom(MP4CompositeAtom):
    pass


class MP4EditListAtom(MP4Atom):
    pass


class MP4MediaAtom(MP4CompositeAtom):
    pass


class MP4MediaHeaderAtom(MP4Atom):
    pass


class MP4HandlerReferenceAtom(MP4Atom):
    pass


class MP4MediaInformationAtom(MP4CompositeAtom):
    pass


class MP4VideoHeaderAtom(MP4Atom):
    pass


class MP4DataInformationAtom(MP4CompositeAtom):
    pass


class MP4DataReferenceAtom(MP4Atom):
    pass


class MP4DataUrlAtom(MP4Atom):
    pass


class MP4SampleTableAtom(MP4CompositeAtom):
    pass


class MP4SampleDescriptionAtom(MP4Atom):
    def __init__(self, file, type, size, metadata=None, recurse=True, required_atoms=None):
        super(MP4SampleDescriptionAtom, self).__init__(file, type, size, materialize=False, seek=False, recurse=recurse, required_atoms=required_atoms)

        self.data = file.read(1 + 3 + 4)
        self.metadata = (BitStream(MP4SampleDescriptionAtom, self.data)
                         .collect_unsigned_int(1, 'version')
                         .collect_string(3, 'flags')
                         .collect_unsigned_int(4, 'sample_count')
                         .value)

        self.children = []
        for _ in range(self.metadata.sample_count):
            atom = MP4Atom.parse(file, recurse=recurse, required_atoms=required_atoms)
            self.children.append(atom)


class MP4AVC1Box(MP4Atom):
    def __init__(self, file, type, size, metadata=None, recurse=True, required_atoms=None):
        super(MP4AVC1Box, self).__init__(file, type, size, seek=False, recurse=recurse, required_atoms=required_atoms)

        header_size = 6 + 2 + 2 + 2 + 12 + 2 + 2 + 4 + 4 + 4 + 2 + 32 + 2 + 2
        self.data = file.read(header_size)
        self.metadata = (BitStream(MP4AVC1Box, self.data)
                         .skip_bits(6 * 8)  # reserved
                         .skip_bits(16)  # data_reference_index
                         .skip_bits(16)  # pre_defined
                         .skip_bits(16)  # reserved
                         .skip_bits(32 * 3)  # pre_defined
                         .collect_unsigned_int(2, 'width')
                         .collect_unsigned_int(2, 'height')
                         .collect_unsigned_int(4, 'horizontal_resolution')
                         .collect_unsigned_int(4, 'vertical_resolution')
                         .skip_bits(32)  # reserved
                         .skip_bits(16)  # frame_count
                         .skip_bits(32 * 8)  # compressor_name
                         .skip_bits(16)  # depth
                         .skip_bits(16)  # pre_defined
                         .value)
        #self.children = [MP4Atom.parse(file)]

        self.children = MP4CompositeAtom.parse_children(file, self.size - header_size, recurse, required_atoms)

class MP4AVCCBox(MP4Atom):
    def __init__(self, file, type, size, recurse=True, required_atoms=None):
        super(MP4AVCCBox, self).__init__(file, type, size, seek=False, recurse=recurse, required_atoms=required_atoms)

        header_size = 4 + 1 + 1
        self.data = file.read(header_size)
        self.metadata = (BitStream(MP4AVCCBox, self.data)
                  .collect_unsigned_int(1, 'version')
                  .collect_unsigned_int(1, 'profile')
                  .collect_unsigned_int(1, 'compatibility')
                  .collect_unsigned_int(1, 'level')
                  .collect_bits(6, 'nal_padding', expected=0b111111)
                  .collect_bits(2, 'nal_length')
                  .collect_bits(3, 'sps_padding', expected=0b111)
                  .collect_bits(5, 'sps_count'))

        # Read counts
        #self.data.append(file.read(4 * stream.sps_count + 4 * stream.pps_count))

        self.sps = []
        for _ in range(self.metadata.sps_count):
            sps_size = int.from_bytes(file.read(2), 'big', signed=False)
            self.sps.append(file.read(sps_size))
            #stream.collect_bits(16, 'sps_size')
            #self.data.append(file.read(stream.sps_size))
            #self.sps.append(stream.collect_string(stream.sps_size, 'sps').sps)

        self.pps = []
        self.metadata.pps_count = int.from_bytes(file.read(1), 'big', signed=False)

        for _ in range(self.metadata.pps_count):
            pps_size = int.from_bytes(file.read(2), 'big', signed=False)
            self.pps.append(file.read(pps_size))
            #stream.collect_bits(16, 'pps_size')
            #self.data.append(file.read(stream.pps_size))
            #self.pps.append(stream.collect_string(stream.pps_size, 'pps').pps)

        #self.metadata = stream.value


class MP4ABox(MP4Atom):
    pass


class MP4SampleTimeAtom(MP4Atom):
    pass


class MP4SyncSampleAtom(MP4Atom):
    def __init__(self, file, type, size, metadata=None, recurse=True, required_atoms=None):
        super(MP4SyncSampleAtom, self).__init__(file, type, size, materialize=False, seek=False, recurse=recurse, required_atoms=required_atoms)

        self.data = file.read(1 + 3 + 4)
        self.metadata = (BitStream(MP4SampleDescriptionAtom, self.data)
                         .collect_unsigned_int(1, 'version')
                         .collect_string(3, 'flags')
                         .collect_unsigned_int(4, 'sample_count')
                         .value)

        self.sizes = [s - 1 for s in struct.unpack_from('>' + 'I' * self.metadata.sample_count,
                                        file.read(4 * self.metadata.sample_count))]
        self.gop_size = self.sizes[1] - self.sizes[0] if \
            len(self.sizes) > 1 and \
            all(s[1] - s[0] == self.sizes[1] - self.sizes[0] for s in zip(self.sizes, self.sizes[1:])) else None


class MP4CompositionTimeAtom(MP4Atom):
    pass


class MP4SampleChunkAtom(MP4Atom):
    pass


class MP4SampleSizeAtom(MP4Atom):
    def __init__(self, file, type, size, metadata=None, recurse=True, required_atoms=None):
        super(MP4SampleSizeAtom, self).__init__(file, type, size, seek=False, materialize=False, recurse=recurse, required_atoms=required_atoms)

        header_size = 12
        self.data = file.read(header_size)
        self.metadata = (BitStream(MP4SampleSizeAtom, self.data)
                         .collect_unsigned_int(1, 'version')
                         .collect_string(3, 'flags')
                         .collect_unsigned_int(4, 'constant_size')
                         .collect_unsigned_int(4, 'size_count')
                         .value)

        self.sizes = struct.unpack_from('>' + 'I' * self.metadata.size_count,
                                        file.read(4 * self.metadata.size_count))
        self.offsets = [0] + list(accumulate(self.sizes)) #(sum, self.sizes, 0)


class MP4SampleOffsetAtom(MP4Atom):
    pass


class MP4MovieExtensionAtom(MP4CompositeAtom):
    pass


class MP4TrackExtensionAtom(MP4Atom):
    pass


class MP4UserDataAtom(MP4Atom):
    pass


class MP4MetadataAtom(MP4Atom):
    pass


class MP4MovieFragmentHeader(MP4Atom):
    pass


class MP4TrackFragmentAtom(MP4CompositeAtom):
    pass


class MP4TrackFragmentHeaderAtom(MP4Atom):
    pass


class MP4TrackFragmentDecodeTimeAtom(MP4Atom):
    pass


class MP4TrackFragmentRunAtom(MP4Atom):
    pass


class MP4TrackFragmentRandomAccessAtom(MP4Atom):
    pass


class MP4MovieFragmentRandomAccessOffsetAtom(MP4Atom):
    pass


class MP4FreeAtom(MP4Atom):
    pass


class MP4SoundHeaderAtom(MP4Atom):
    pass

types = {
    b'ftyp': MP4FileTypeAtom,

    b'moov': MP4MovieAtom,
    b'mvhd': MP4MoveHeaderAtom,
    b'trak': MP4TrackAtom,
    b'tkhd': MP4TrackHeaderAtom,
    b'edts': MP4EditAtom,
    b'elst': MP4EditListAtom,
    b'mdia': MP4MediaAtom,
    b'mdhd': MP4MediaHeaderAtom,
    b'hdlr': MP4HandlerReferenceAtom,
    b'minf': MP4MediaInformationAtom,
    b'vmhd': MP4VideoHeaderAtom,
    b'dinf': MP4DataInformationAtom,
    b'dref': MP4DataReferenceAtom,
    b'stbl': MP4SampleTableAtom,
    b'stsd': MP4SampleDescriptionAtom,
    b'avc1': MP4AVC1Box,
    b'avcC': MP4AVCCBox,
    b'mp4a': MP4ABox,
    b'stts': MP4SampleTimeAtom,
    b'stsc': MP4SampleChunkAtom,
    b'stsz': MP4SampleSizeAtom,
    b'stco': MP4SampleOffsetAtom,
    b'mvex': MP4MovieExtensionAtom,
    b'trex': MP4TrackExtensionAtom,
    b'udta': MP4UserDataAtom,
    b'meta': MP4MetadataAtom,
    b'smhd': MP4SoundHeaderAtom,

    b'moof': MP4MovieFormatAtom,
    b'mfhd': MP4MovieFragmentHeader,
    b'traf': MP4TrackFragmentAtom,
    b'tfhd': MP4TrackFragmentHeaderAtom,
    b'tfdt': MP4TrackFragmentDecodeTimeAtom,
    b'trun': MP4TrackFragmentRunAtom,

    b'mdat': MP4MovieDataAtom,

    b'mfra': MP4MovieFragmentRandomAccessAtom,
    b'tfra': MP4TrackFragmentRandomAccessAtom,
    b'mfro': MP4MovieFragmentRandomAccessOffsetAtom,
    b'free': MP4FreeAtom,

    b'stss': MP4SyncSampleAtom,
    b'ctts': MP4CompositionTimeAtom
}
