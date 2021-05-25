from collections import namedtuple
from functools import reduce


class BitStream(object):
    _namedtuple_cache = {}

    def __init__(self, type, data, start=0):
        self.type = type
        self.data = data
        self.index = start * 8
        self.values = {'start': self.index}

    # TODO switch functions to use self and not self.values
    def __getattr__(self, key):
        return self.values[key]

    def skip_true(self):
        bit = self._next_bit()
        assert bit
        return self

    def skip_false(self):
        bit = self._next_bit()
        assert not bit
        return self

    def skip_bits(self, n, boolean=None):
        if boolean is None or boolean:
            self.index += n
        return self

    def skip_bits_f(self, f):
        self.skip_bits(f(self.values))
        return self

    def skip_exponential_golumb(self, n=1, f=None):
        if f is not None:
            n = f(self.values)
        while n > 0:
            self.skip_bits(self._get_exponential_golumb_size())
            n -= 1
        return self

    def skip_exponential_golumb_if(self, key=None, f=None, boolean=None, n=1):
        if ((key is not None and self.values[key]) or
                (f is not None and f(self.values)) or
                (boolean is not None and boolean)):
            self.skip_exponential_golumb(n=n)
        return self

    def collect_bit(self, name=None, expected=None):
        bit = self._next_bit()
        if expected is not None:
            assert bit == expected
        self.values[name or len(self.values)] = bit
        return self

    def collect_bits(self, n, name, expected=None):
        self.values[name or len(self.values)] = self._next_bits(n)
        if expected is not None:
            assert self.values[name or len(self.values)] == expected
        return self

    def collect_unsigned_exponential_golumb(self, name=None):
        self.values[name or len(self.values)] = self.get_unsigned_exponential_golumb()
        return self

    def collect_exponential_golumb_if(self, name, key):
        if self.values[key]:
            self.collect_unsigned_exponential_golumb(name)
        return self

    # TODO redundant, collect_bits does the same
    def collect_unsigned_int(self, byte_size, name=None, condition=None):
        assert  self.index % 8 == 0

        value = 0
        for v in self.data[self.index // 8:self.index // 8 + byte_size]:
            value = (value << 8) | v
        self.index += byte_size * 8
        self.values[name or len(self.values)] = value
        return self

    # TODO redundant, collect_bits does the same
    def collect_unsigned_int_if(self, byte_size, name, condition):
        if condition(self.values):
            self.collect_unsigned_int(byte_size, name)
        return self

    def collect_string(self, byte_size, name=None):
        assert self.index % 8 == 0

        value = str(self.data[self.index // 8:self.index // 8 + byte_size])
        self.values[name or len(self.values)] = value
        self.index += byte_size * 8
        return self

    def collect_string_f(self, byte_size, f, name=None):
        assert self.index % 8 == 0

        values = []
        size = byte_size * f(self.values)

        for index in range(self.index // 8, self.index // 8 + size, byte_size):
            values.append(str(self.data[index:index + byte_size]))

        self.index += size * 8
        self.values[name or len(self.values)] = values

        return self

    def mark_position(self, name=None):
        self.values[name or len(self.values)] = self.index
        return self

    def byte_align(self, expected=None):
        value = self._next_bits((8 - self.index % 8) % 8)
        if expected is not None:
            assert value == expected
        return self

    def skip_entry_point_offsets(self):
        num_entry_point_offsets = self.get_unsigned_exponential_golumb()
        if num_entry_point_offsets:
            offset_len_minus1 = self.get_unsigned_exponential_golumb()
            self.skip_bits((offset_len_minus1 + 1) * num_entry_point_offsets)
        return self

    def skip_entry_point_offsets_if(self, boolean):
        if boolean:
            self.skip_entry_point_offsets()
        return self

    @property
    def value(self):
        self.values['end'] = self.index
        if self.type not in self._namedtuple_cache:
            self._namedtuple_cache[self.type] = namedtuple(self.type.__name__ + 'BitStream', self.values.keys())
        return self._namedtuple_cache[self.type](**self.values)

    def _next_bit(self):
        value = bool(self.data[self.index >> 3] & (128 >> (self.index % 8)))
        self.skip_bits(1)
        return value

    def _next_bits(self, n):
        return reduce(lambda a, _: (a << 1) | self._next_bit(), range(n), 0)

    def get_unsigned_exponential_golumb(self):
        size = self._get_exponential_golumb_size()
        return (1 << size | self._next_bits(size)) - 1

    def _get_exponential_golumb_size(self):
        size = 0
        while self._next_bit() == 0:
            size += 1
        return size
