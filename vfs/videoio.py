import math
import os
import numpy as np
import ffmpeg

H264 = "h264"
HEVC = "hevc"
RGB8 = "rgb"

extensions = {
    H264: "h264",
    HEVC: "hevc",
    RGB8: "rgb"
}

DEFAULT_PIXEL_FORMAT = 'yuv420p'
pixel_formats = {
    RGB8: "rgb8"
}

bits_per_pixel = {
    RGB8: 24
}

encoded = {
    H264: True,
    HEVC: True,
    RGB8: False
}
encoders = {
    H264: "nvenc_h264",
    HEVC: "nvenc_hevc",
    RGB8: "rawvideo"
}
decoders = {
    H264: "h264_cuvid",
    HEVC: "hevc_cuvid"
}

RAW_VIDEO_MAX_GOP_SIZE = (30*3840*2160*bits_per_pixel[RGB8]) // 8


class VideoReader(object):
    def __init__(self, filename, shape, codec, limit=None, loglevel='error'):
        self.shape = shape
        self.eof = False
        self._buffer = bytearray(np.prod(shape))

        if encoded[codec]:
            self._stream = (ffmpeg
                    .input(filename, codec=decoders[codec])  #'h264_cuvid')
                    .output('pipe:', format='rawvideo', pix_fmt='rgb8', loglevel=loglevel,
                            **dict(vframes=limit) if limit else {})
                    .run_async(pipe_stdout=True))
        else:
            self._stream = (ffmpeg
                            .input(filename,
                                   format='rawvideo',
                                   pix_fmt=pixel_formats.get(codec, None),
                                   s=_size(shape))
                            .output('pipe:', format='rawvideo', pix_fmt='rgb8', loglevel=loglevel,
                                    **dict(vframes=limit) if limit else {})
                            .run_async(pipe_stdout=True))

        self._frame = (np
                    .frombuffer(self._buffer, np.uint8)
                    .reshape(self.shape))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def read(self):
        if self._stream.stdout.readinto(self._buffer):
            return self._frame
        else:
            self.eof = True
            return None

    def close(self):
        self._stream.stdout.close()
        self._stream.wait()


class NullReader(object):
    def __enter__(self):
        self.eof = True
        return self

    def __exit__(self, *args):
        pass

    def read(self, *args):
        return None

class VideoWriter(object):
    def __init__(self, filename, shape, codec=H264, loglevel='error'):
        self.shape = shape
        self._stream = (ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb8',
                   s=_size(shape), loglevel=loglevel)
            .output(filename, pix_fmt='yuv420p', codec=encoders[codec]) #'nvenc_h264')
            .overwrite_output()
            .run_async(pipe_stdin=True))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, frame):
        self._stream.stdin.write(frame.tobytes())

    def close(self):
        self._stream.stdin.close()
        self._stream.wait()


class NullWriter(object):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def write(self, *args):
        pass

def _size(resolution):
    return '{}x{}'.format(resolution[1], resolution[0])

def _crop(roi):
    return (roi[1], roi[0], roi[3] - roi[1], roi[2] - roi[0])

def _psnr_to_mse(psnr):
    # PSNR = 10*log10(MAX^2/MSE)
    return 255 ** 2 / 10 ** (psnr / 10)

def _get_psnr_from_stderr(stderr):
    return float(str(stderr[1]).split('\\n')[-2].split('max:')[-1])

def split_video(source_filename, output_filename_template, resolution, codec, fps, loglevel='error'):
    if encoded[codec]:
        return split_encoded_video(source_filename, output_filename_template, codec, loglevel)
    else:
        return split_raw_video(source_filename, output_filename_template, resolution, codec, fps, loglevel=loglevel)

def split_encoded_video(source_filename, output_filename_template, codec=None, loglevel='error'):
    path = os.path.dirname(output_filename_template)
    stream = (ffmpeg
     .input(source_filename, codec=decoders[codec])
     .output(output_filename_template, format='segment', segment_list='pipe:', segment_list_type='csv',
             segment_time=1, vcodec='copy', loglevel=loglevel)
     .run_async(pipe_stdout=True))

#    return (os.path.join(path, filename)
#            for filename in stream.stdout.raw.readall().decode('utf-8').strip().split('\n'))
    return ((os.path.join(path, filename), float(start_time), float(end_time))
            for (filename, start_time, end_time)
            in map(lambda row: row.split(','),
                   stream.stdout.raw.readall().decode('utf-8').strip().split('\n')))

def split_raw_video(source_filename, output_filename_template, resolution, codec, fps, segment_time=None, loglevel='error'):
    path = os.path.dirname(output_filename_template)
    frame_size = (bits_per_pixel[codec] * np.prod(resolution)) // 8
    segment_time = segment_time or max(RAW_VIDEO_MAX_GOP_SIZE // frame_size, 1) / fps
    stream = (ffmpeg
     .input(source_filename, f='rawvideo', pix_fmt=pixel_formats[codec], s=_size(resolution))
     .output(output_filename_template,
             format='segment',
             segment_list='pipe:',
             segment_time=segment_time,
             segment_list_type='csv',
             vcodec='copy',
             loglevel=loglevel)
     .run_async(pipe_stdout=True))

    return ((os.path.join(path, filename), float(start_time), float(end_time))
            for (filename, start_time, end_time)
            in map(lambda row: row.split(','),
                   stream.stdout.raw.readall().decode('utf-8').strip().split('\n')))

def join_video(input_filenames, output_filename, resolution, codec, loglevel='error'):
    if encoded[codec]:
        join_encoded(input_filenames, output_filename, loglevel=loglevel)
    else:
        join_raw(input_filenames, output_filename, resolution, codec, loglevel=loglevel)

def join_encoded(input_filenames, output_filename, loglevel='error'):
    (ffmpeg
      .input('concat:{}'.format('|'.join(input_filenames)))
      .output(output_filename, c='copy', loglevel=loglevel)
      .overwrite_output()
      .run())

def join_raw(input_filenames, output_filename, resolution, codec, loglevel='error'):
    (ffmpeg
      .input('concat:{}'.format('|'.join(input_filenames)),
             s=_size(resolution),
             pix_fmt=pixel_formats[codec])
      .output(output_filename, c='copy', loglevel=loglevel)
      .overwrite_output()
      .run())

def reformat(input_filename, output_filename, output_resolution, output_codec, output_fps,
             input_resolution=None, input_codec=None, input_fps=None, roi=None, times=None, loglevel='error'):
    if encoded[input_codec]:
        return reformat_encoded(input_filename, output_filename,
                                input_resolution, input_fps, output_resolution, output_codec, roi, times, output_fps, loglevel=loglevel)
    else:
        return reformat_raw(input_filename, output_filename, input_resolution, input_fps, output_resolution,
                            input_codec, output_codec, roi, times, output_fps, loglevel=loglevel)

def reformat_encoded(input_filename, output_filename, input_resolution, input_fps, output_resolution, codec, roi, times,
                     output_fps, loglevel='error'):
    assert(times is None or times[1] - times[0] > 0)

    if not output_fps:
        output_fps = input_fps

    op = ffmpeg.input(input_filename, r=input_fps, codec=decoders[codec])

    if roi is not None and roi != (0, 0, *input_resolution): #*output_resolution):
        op = op.crop(*_crop(roi))
    if input_resolution != output_resolution:
        op = op.filter('scale', output_resolution[1], output_resolution[0])

    if times:
        (op
            .output(output_filename, codec=encoders[codec], ss=times[0], to=times[1], r=output_fps, loglevel=loglevel)
            .run())
        return 0
    else:
        (op
            .output(output_filename, codec=encoders[codec], r=output_fps, loglevel=loglevel)
            .run())
        return 0

def reformat_raw(input_filename, output_filename, input_resolution, input_fps, output_resolution, input_codec, output_codec, roi,
                 times, output_fps, loglevel='error'):
    if not output_fps:
        output_fps = input_fps

    if times is not None:
        op = ffmpeg.input(input_filename,
                    format='rawvideo',
                    pix_fmt=pixel_formats[input_codec],
                    s=_size(input_resolution),
                    ss=times[0],
                    to=times[1],
                    r=input_fps)
    else:
        op = ffmpeg.input(input_filename,
                    format='rawvideo',
                    pix_fmt=pixel_formats[input_codec],
                    s=_size(input_resolution),
                    r=input_fps)

    if roi is not None and roi != (0, 0, *input_resolution): #*output_resolution):
        op = op.crop(*_crop(roi))
    if input_resolution != output_resolution or input_fps != output_fps:
        pretransform = op.split() #.filter_multi_output('split')

        if input_resolution != output_resolution:
            op = pretransform[0].filter('scale', output_resolution[1], output_resolution[0])
        if input_fps != output_fps:
            op = op.filter('fps', input_fps)

        posttransform = op.split()
        error = posttransform[1]

        if input_resolution != output_resolution:
            if roi is not None:
                error = error.filter('scale', roi[3] - roi[1], roi[2] - roi[0])
            else:
                error = error.filter('scale', input_resolution)
        if input_fps != output_fps:
            error = error.filter('fps', input_fps)

        error = ffmpeg.filter([pretransform[1], error], 'psnr')

        if output_codec in pixel_formats:
            result_out = posttransform[0].output(output_filename, codec=encoders[output_codec], pix_fmt=pixel_formats[output_codec])
        else:
            result_out = posttransform[0].output(output_filename, codec=encoders[output_codec], pix_fmt=DEFAULT_PIXEL_FORMAT)
        error_out = error.output('-', f='null', codec=encoders[output_codec], hide_banner=None, nostats=None) #, loglevel=loglevel)

        stderr = ffmpeg.run([result_out, error_out], capture_stderr=True)
        psnr = _get_psnr_from_stderr(stderr)
        return _psnr_to_mse(psnr)
    else:
        (op
             .output(output_filename, codec=encoders[output_codec], r=output_fps, loglevel=loglevel)
             .run())
        return 0

def get_shape(filename):
    probe = ffmpeg.probe(filename)
    assert(len(probe['streams']) == 1)

    stream = probe['streams'][0]
    return stream['height'], stream['width']

def get_shape_and_codec(filename):
    probe = ffmpeg.probe(filename)
    assert(len(probe['streams']) == 1)

    stream = probe['streams'][0]
    fps_ratio = list(map(int, stream['r_frame_rate'].split('/')))
    return (stream['height'], stream['width']), stream['codec_name'], fps_ratio[0] / fps_ratio[1]

def compute_mse(reference_gop, target_gop):
    if encoded[reference_gop.video().codec]:
        reference = ffmpeg.input(reference_gop.filename, codec=decoders[reference_gop.video().codec], r=reference_gop.fps)
    else:
        reference = ffmpeg.input(reference_gop.filename,
                                 format='rawvideo',
                                 pix_fmt=pixel_formats[reference_gop.video().codec],
                                 s=_size(reference_gop.video().shape()),
                                 r=reference_gop.fps)

    if encoded[target_gop.video().codec]:
        target = ffmpeg.input(target_gop.filename, codec=decoders[target_gop.video().codec], r=target_gop.fps)
    else:
        target = ffmpeg.input(target_gop.filename,
                                 format='rawvideo',
                                 pix_fmt=pixel_formats[target_gop.video().codec],
                                 s=_size(target_gop.video().shape()),
                                 r=target_gop.fps)

    if reference_gop.video().shape() == target_gop.video().shape() and \
        reference_gop.fps == target_gop.fps:
        filter = ffmpeg.filter([reference, target], 'psnr')
    else:
        filter = (target.filter('scale', reference_gop.video().width, reference_gop.video().height)
                        .filter('fps', reference_gop.fps))
        filter = ffmpeg.filter([reference, filter], 'psnr')

    output = filter.output('-', f='null', hide_banner=None, nostats=None)

    stderr = output.run(capture_stderr=True)
    psnr = _get_psnr_from_stderr(stderr)
    return _psnr_to_mse(psnr)
