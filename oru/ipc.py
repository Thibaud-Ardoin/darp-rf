import os
from collections import defaultdict,deque
import io
import subprocess
import time
import struct
import dataclasses
import re
from typing import Tuple
STRUCT_CHAR_SET = set('xcbB?hHiIlLqQnNefdspP')


class StructStreamParser:
    STREAM_HEADER_RE = re.compile(r'^(?P<type>[xcbB?hHiIlLqQnNefdspP])(?P<quantifier>(\d+|\*))?')

    def __init__(self, stream_with_header: io.BufferedIOBase):
        packet_header, packet_fixedlen, packet_varlen = self._parse_stream_header(stream_with_header)
        self.packet_h_fmt = struct.Struct('=' + packet_header)
        self.packet_fl_fmt = struct.Struct('=' + packet_fixedlen)
        self.packet_vl_types = packet_varlen
        self.packet_vl_count = len(packet_varlen)
        self.packet_vl_type_sizes = list(map(struct.calcsize, packet_varlen))

    def __call__(self, stream: io.BufferedIOBase):
        data = stream.read()
        if data is None:
            return
        offset = 0
        while offset < len(data):
            packet_vl_lengths = self.packet_h_fmt.unpack(data[offset:offset+self.packet_h_fmt.size])
            offset += self.packet_h_fmt.size
            fl_data = self.packet_fl_fmt.unpack(data[offset:offset+self.packet_fl_fmt.size])
            offset += self.packet_fl_fmt.size
            vl_data = [None] * self.packet_vl_count
            for idx, (t, s, n) in enumerate(zip(self.packet_vl_types, self.packet_vl_type_sizes, packet_vl_lengths)):
                sz = s*n
                vl_data[idx] = struct.unpack('=' + t * n, data[offset:offset+sz])
                offset += sz

            yield fl_data + tuple(vl_data)

    def _parse_stream_header(self, stream):
        first_byte = None
        while first_byte is None:
            first_byte = stream.read(1)

        stream_header_length = first_byte[0]
        stream_format_string = stream.read(stream_header_length - 1).decode()
        original_header = stream_format_string
        stream_packet_header_fmt = ''
        stream_packet_fixedlen_types = ''
        stream_packet_varlen_types = ''

        while len(stream_format_string) > 0:
            m = self.STREAM_HEADER_RE.match(stream_format_string)
            if m is None:
                raise IOError(f"Bad stream header: `{stream_format_string[0]}` is not a valid type character "
                              f"(read header `{original_header}` from stream). Valid type chars are " +
                              str(STRUCT_CHAR_SET))
            stream_format_string = stream_format_string[len(m.group(0)):]
            m = m.groupdict()
            typechar = m['type']
            if m['quantifier'] == '*':
                stream_packet_varlen_types += typechar
                stream_packet_header_fmt += 'B'
            elif len(stream_packet_header_fmt) > 0:
                raise IOError("Bad stream header: variable length types must be on the end of the header fmt string.")
            else:
                if m['quantifier'] is not None:
                    typechar *= int(m['quantifier'])
                stream_packet_fixedlen_types += typechar

        return stream_packet_header_fmt, stream_packet_fixedlen_types, stream_packet_varlen_types


class PipeReader:
    fp : io.BufferedReader
    def __init__(self, proc, fd : int):
        self.fp = open(fd, 'rb')
        self.proc = proc

    def read(self, n):
        buf = memoryview(bytearray(n))
        m = 0
        while m < n:
            k = self.fp.readinto(buf[m:])
            if k is None:
                if self.proc.poll() is not None:
                    break
            else:
                m += k

        buf = bytes(buf[:m])
        return buf

    def close(self):
        self.fp.close()

def nonblocking_subprocess_pipe():
    (p_r, p_w) = os.pipe()
    os.set_blocking(p_r, False)
    return p_r, p_w
