import datetime
import re
from fractions import Fraction
import numpy as np

import vfs

from vfs.logicalvideo import LogicalVideo
from vfs.videoio import encoded, channels

resolutions = {'4K': (2160, 3840), '2K': (1080, 1920), '1K': (540, 960)}

def load(name):
    return Array(name)

class UnitFraction(Fraction):
    def __new__(cls, *args, units=None, format=None, **kwargs):
        instance = super().__new__(cls, *args)
        instance.units = units
        instance.format = format
        return instance

    def __eq__(self, other):
        return super().__eq__(other) and (not isinstance(other, UnitFraction) or self.units == other.units)

    def __repr__(self):
        return f'{super().__repr__()}' + f'({self.units})' if self.units else ''

    def __str__(self):
        if self.format is None:
            return super().__str__() + (self.units or '')
        elif isinstance(self.format, str):
            return ('{value:%s}' % self.format).format(value=float(self)) + (self.units or '')
        else:
            return self.format(self)

    def _unary_operator(f):
        return lambda self: UnitFraction(f(self), units=self.units, format=self.format)

    def _binary_operator(f):
        def op(left, right):
            if right is None:
                assert (False)
            #if isinstance(right, UnitFraction) and left.units != right.units:
            #    raise ArithmeticError()
            #else:
            return UnitFraction(f(left, right), units=left.units, format=left.format)
        return op

    __add__ = _binary_operator(Fraction.__add__)
    __mul__ = _binary_operator(Fraction.__mul__)
    __radd__ = _binary_operator(Fraction.__radd__)
    __rmul__ = _binary_operator(Fraction.__rmul__)
    __rsub__ = _binary_operator(Fraction.__rsub__)
    __rtruediv__ = _binary_operator(Fraction.__rtruediv__)
    __sub__ = _binary_operator(Fraction.__sub__)
    __truediv__ = _binary_operator(Fraction.__truediv__)
    __floordiv__ = _binary_operator(Fraction.__floordiv__)
    __rfloordiv__ = _binary_operator(Fraction.__rfloordiv__)
    __mod__ = _binary_operator(Fraction.__mod__)
    __rmod__ = _binary_operator(Fraction.__rmod__)
    __pow__ = _binary_operator(Fraction.__pow__)
    __rpow__ = _binary_operator(Fraction.__rpow__)

    __pos__ = _unary_operator(Fraction.__pos__)
    __neg__ = _unary_operator(Fraction.__neg__)
    __abs__ = _unary_operator(Fraction.__abs__)
    __floor__ = _unary_operator(Fraction.__floor__)
    __ceil__ = _unary_operator(Fraction.__ceil__)
    __round__ = _unary_operator(Fraction.__round__)
    __rdivmod__ = _unary_operator(Fraction.__rdivmod__)
    __divmod__ = _unary_operator(Fraction.__divmod__)

"""
class Percentage(UnitFraction):
    def __new__(cls, value, *args, format=None, clamp=True, **kwargs):
        instance = super().__new__(cls, value/100, *args, format=format or (lambda v: f'{float(v)*100:g}%'), **kwargs)

        if clamp and instance > 1:
            return cls.__new__(cls, 1, clamp=clamp, **kwargs)
        elif clamp and instance < 0:
            return cls.__new__(cls, 0, clamp=clamp, **kwargs)

        instance.clamp = clamp
        return instance

    def _unary_operator(f):
        return lambda self: Percentage(f(self), units=self.units, clamp=self.clamp, format=self.format)

    def _binary_operator(f):
        def op(left, right):
            return Percentage(f(left, right), units=left.units, clamp=left.clamp or right.clamp, format=left.format)
        return op

    __add__ = _binary_operator(UnitFraction.__add__)
    __mul__ = _binary_operator(UnitFraction.__mul__)
    __radd__ = _binary_operator(UnitFraction.__radd__)
    __rmul__ = _binary_operator(UnitFraction.__rmul__)
    __rsub__ = _binary_operator(UnitFraction.__rsub__)
    __rtruediv__ = _binary_operator(UnitFraction.__rtruediv__)
    __sub__ = _binary_operator(UnitFraction.__sub__)
    __truediv__ = _binary_operator(UnitFraction.__truediv__)
    __floordiv__ = _binary_operator(UnitFraction.__floordiv__)
    __rfloordiv__ = _binary_operator(UnitFraction.__rfloordiv__)
    __mod__ = _binary_operator(UnitFraction.__mod__)
    __rmod__ = _binary_operator(UnitFraction.__rmod__)
    __pow__ = _binary_operator(UnitFraction.__pow__)
    __rpow__ = _binary_operator(UnitFraction.__rpow__)

    __pos__ = _unary_operator(UnitFraction.__pos__)
    __neg__ = _unary_operator(UnitFraction.__neg__)
    __abs__ = _unary_operator(UnitFraction.__abs__)
    __floor__ = _unary_operator(UnitFraction.__floor__)
    __ceil__ = _unary_operator(UnitFraction.__ceil__)
    __round__ = _unary_operator(UnitFraction.__round__)
    __rdivmod__ = _unary_operator(UnitFraction.__rdivmod__)
    __divmod__ = _unary_operator(UnitFraction.__divmod__)
"""

class InvertedRepresentationUnitFraction(UnitFraction):
    def __new__(cls, *args, units=None, inverted_units=None, **kwargs):
        instance = super().__new__(cls, *args, units=units, **kwargs)
        instance.inverted_units = inverted_units
        return instance

    def __str__(self):
        return Fraction.__str__(1 / self) + (self.inverted_units or '')
        #if self.denominator == 1:
        #    return f'{self.numerator}'

    def _unary_operator(f):
        return lambda self: InvertedRepresentationUnitFraction(
            f(self), units=self.units, inverted_units=self.inverted_units, format=self.format)

    def _binary_operator(f):
        def op(left, right=None):
            return InvertedRepresentationUnitFraction(f(left, right),
                                                      units=left.units,
                                                      inverted_units=left.inverted_units,
                                                      format=left.format)

        return op

    __add__ = _binary_operator(UnitFraction.__add__)
    __mul__ = _binary_operator(UnitFraction.__mul__)
    __radd__ = _binary_operator(UnitFraction.__radd__)
    __rmul__ = _binary_operator(UnitFraction.__rmul__)
    __rsub__ = _binary_operator(UnitFraction.__rsub__)
    __rtruediv__ = _binary_operator(UnitFraction.__rtruediv__)
    __sub__ = _binary_operator(UnitFraction.__sub__)
    __truediv__ = _binary_operator(UnitFraction.__truediv__)
    __floordiv__ = _binary_operator(UnitFraction.__floordiv__)
    __rfloordiv__ = _binary_operator(UnitFraction.__rfloordiv__)
    __mod__ = _binary_operator(UnitFraction.__mod__)
    __rmod__ = _binary_operator(UnitFraction.__rmod__)
    __pow__ = _binary_operator(UnitFraction.__pow__)
    __rpow__ = _binary_operator(UnitFraction.__rpow__)
    __pos__ = _binary_operator(UnitFraction.__pos__)
    __neg__ = _binary_operator(UnitFraction.__neg__)
    __abs__ = _binary_operator(UnitFraction.__abs__)

    __floor__ = _unary_operator(UnitFraction.__floor__)
    __ceil__ = _unary_operator(UnitFraction.__ceil__)
    __round__ = _unary_operator(UnitFraction.__round__)
    __rdivmod__ = _unary_operator(UnitFraction.__rdivmod__)
    __divmod__ = _unary_operator(UnitFraction.__divmod__)

#def percentage(*args, format=None, clamp=True, **kwargs):
#    return Percentage(*args, format=format, clamp=clamp, **kwargs)


def spf(*args, **kwargs):
    return InvertedRepresentationUnitFraction(*args, inverted_units='fps', **kwargs)

def seconds(*args, format=None, **kwargs):
    return UnitFraction(*args, units='s', format=format or _to_timedelta_string, **kwargs)

def _to_timedelta_string(value, precision=2):
    repr = str(datetime.timedelta(seconds=float(value)))
    repr = re.sub(r'(\d+):(\d+):(\d+\.?\d{,%d})(\d*)?' % precision, r'\1h\2m\3s', repr) # 01:01 -> 01h:01m
    repr = re.sub(r'(^0+h)|(?<!0)(0{2}[ms])', '', repr) # 00m05s -> 05s
    repr = re.sub(r'(?:^|(?<=[hm]))0+([^0]+(?:\.?\d+)?[ms])', r'\1', repr) # 05m -> 5m
    return repr

class Array:
    # TODO change to new, make these immutable
    def __init__(self, name_or_array, x=None, y=None, t=None, shape=None):
        if isinstance(name_or_array, str):
            name = name_or_array
            self._shape = None

            if not LogicalVideo.exists_by_name(name):
                raise KeyError()
            else:
                self.video = LogicalVideo.get_by_name(name)
                self._shape = tuple([self._get_duration()] + list(self._get_resolution()))
                self._y = slice(0, self.shape[1], 1) #slice(percentage(0), percentage(100), 1)
                self._x = slice(0, self.shape[2], 1) #slice(percentage(0), percentage(100), 1)
                self._t = slice(seconds(0), self.shape[0], self._get_spf())
        elif isinstance(name_or_array, Array):
            array = name_or_array

            self.video = array.video
            self._shape = shape if shape else array._shape
            divisors = array._shape[1] / self._shape[1], array._shape[2] / self._shape[2]

            self._t = self._slice(array._t, t, self.shape[0]) #if t and t != slice(None, None, None) else array._t

            #if array._y != slice(Percentage(0), Percentage(100), self._get_spf()):
            self._y = self._slice(array._y, y, self.shape[1], divisors[0], cast=int) #if y and y != slice(None, None, None) else array._y
            #else:
            #    self._y = array._y

            #if isinstance(array._x, Percentage):
            #if isinstance(array._x, Percentage):
            self._x = self._slice(array._x, x, self.shape[2], divisors[1], cast=int) #if x and x != slice(None, None, None) else array._x

            self._shape = (self._t.stop - self._t.start,
                           self._y.stop - self._y.start,
                           self._x.stop - self._x.start)
            #else:
            #    self._x = array._x

            #x0 = array.x[0] + (x[0] or 0)
            #x1 = min(array.x[1] - array.x[0] - x0, x0 + x[1] - x[0])
            #y0 = array.y[0] + (y[0] or 0)
            #y1 = min(array.y[1] - array.y[0] - y0, y0 + y[1] - y[0])
            #t0 = array.t[0] + (t[0] or 0)
            #t1 = min(array.t[1] - array.t[0] - t0, t0 + t[1] - t[0], self.video.duration() - t0)
            #self.x, self.y, self.t = (x0, x1), (y0, y1), (t0, t1)
        else:
            raise IndexError()

    def __getitem__(self, address):
        if not isinstance(address, tuple):
            return self.__getitem__((address, slice(0, self.shape[1], 1), slice(0, self.shape[2], 1)))
        elif isinstance(address, tuple) and len(address) == 1:
            return self.__getitem__(address[0], (slice(0, self.shape[1], 1), slice(0, self.shape[2], 1)))
        elif isinstance(address, tuple) and len(address) == 2:
            return self.__getitem__(address[0], address[1], (slice(0, self.shape[2], 1)))
        elif isinstance(address, tuple) and len(address) != 3:
            raise IndexError()
        elif isinstance(address, tuple):
            t, y, x = address
            return Array(self, t=t, y=y, x=x)
#        elif isinstance(address, int):
#            print(f'int {address}')
#            return self.__getitem__((slice(None, None, None), slice(None, None, None), address))
#        elif isinstance(address, frame):
#            print(f'frame {address}')
        return self

    def __str__(self):
        value = f'{self.video.name}({self.shape[1]}x{self.shape[2]}@{self._get_fps()})[{self._slice_repr(self._t, 0, self._shape[0], default_step=self._get_spf())}, {self._slice_repr(self._y, max=self.shape[1])}, {self._slice_repr(self._x, max=self.shape[2])}]'
        return re.sub(r'(, :)+\]', ']', value).replace('[:]', '')

    #def __repr__(self):
    #    return str(self)

    def at(self, resolution):
        if isinstance(resolution, str):
            return self.at(resolutions[resolution])
        elif isinstance(resolution, tuple) and len(resolution) == 2:
            return Array(self, shape=(self.shape[0], resolution[0], resolution[1]))
        else:
            raise IndexError()

    def to_numpy(self, codec='rgb'):
        with vfs.api.read(
                name=self.video.name,
                filename=None,
                resolution=(self._y.stop - self._y.start, self._x.stop - self._x.start), #self.shape[1:],
                roi=(float(self._y.start), float(self._x.start), float(self._y.stop), float(self._x.stop)),
                t=[float(self._t.start), float(self._t.stop)],
                codec=codec,
                fps=float(1 / self._t.step)) as stream:
            data = np.frombuffer(stream.read(), dtype=np.uint8)

            if not encoded[codec]:
                data = data.reshape(
                    int(self._y.stop - self._y.start),
                    channels[codec],
                    int(self._x.stop - self._x.start),
                    int((self._t.stop - self._t.start) / self._t.step))
            return data

    def array_split(self, indices_or_sections, codec='rgb', axis=0):
        if axis == 0:
            sections = indices_or_sections
            duration = self._t.stop - self._t.start

            if isinstance(indices_or_sections, int):
                partitions = (((duration * i) / sections , (duration * (i + 1)) / sections) for i in range(sections))
            else:
                indices = list(indices_or_sections)
                if len(indices) == 0 or indices[0] != 0: indices.insert(0, 0)
                if indices[-1] != self._t.stop: indices.append(self._t.stop)
                partitions = (((duration * indices[i]) / sections , (duration * indices[i + 1]) / sections) for i in range(len(indices)))
            # TODO This should all be pushed into vfs.readmany
            #p = list(partitions)
            foo = []
            for start, end in partitions:
                print(self[start:end])
                foo.append(self[start:end].to_numpy(codec))
            return foo
            #return (self[start:end].to_numpy(codec) for start, end in partitions)
        else:
            return self.to_numpy(codec).array_split(indices_or_sections, axis=axis)
        pass

    @staticmethod
    def _slice(current, new, max_value, divisor=1, cast=lambda v: v):
        new = new or slice(None, None, None)
        #new = slice(new.start if new and new.start is not None else 0,
        #            new.stop if new and new.stop is not None else max_value,
        #            new.step if new and new.step is not None else current.step)

        #if isinstance(current.start, Percentage):
        #    start = new.start * current.start if new.start is not None else current.start
        #else:
        #    start = new.start + current.start if new.start is not None else current.start

        start = new.start + current.start if new.start is not None else current.start
        #stop = min(current.stop - current.start - start, start + new.stop - (new.start or 0), max_value - start) if new.stop is not None else current.stop
        stop = min(current.stop, start + new.stop - (new.start or 0), max(current.stop, max_value - start)) if new.stop is not None else current.stop
        step = new.step if new.step is not None else current.step #current.step * new.step

        return slice(cast(start / (divisor or 1)), cast(stop / (divisor or 1)), step)

    @property
    def shape(self):
        return self._shape #1,1, self.video.duration()

    def _get_spf(self):
        return 1 / spf(30)

    def _get_fps(self):
        return UnitFraction(1 / self._get_spf(), units="fps")

    def _get_duration(self):
        frames = int(self._shape[0] if self._shape else (self.video.duration() * self._get_fps()))
        return seconds(frames / self._get_fps())

    def _get_resolution(self):
        return self.video.videos()[0].resolution() # resolutions['4K']

    @staticmethod
    def _slice_repr(value, min=None, max=None, default_step=1):
        result = (str(value.start or '') if value.start != min else '') + ':'
        result += str(value.stop or '') if value.stop != max else ''
        result += (':' + str(value.step or '')) if value.step != default_step else ''

        if result.startswith('0%'):
            result = result[2:]
        if result.endswith('100%'):
            result = result[:-4]
        if result.endswith('::'):
            result = result[:-1]

        return result
