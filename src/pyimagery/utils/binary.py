import re
import struct
from collections import deque
from typing import List, Iterable, Literal, Optional, Sequence, Tuple

from pyimagery._type_hint import Bitcode

Intdtype = Literal["b", "h", "l", "q"]
ENDIAN = "big"


def tobin(
    __data: int,
    __dtype: Intdtype,
    bitlength: Optional[int] = None,
    *,
    signed: bool = True,
) -> Bitcode:

    if not isinstance(__data, Iterable):
        __data = [__data]
    if __dtype is int:

        bindata = [None] * len(__data)
        if signed:
            for index, d in enumerate(__data):
                b = bin(d)
                b = "0" + b[2:] if d >= 0 else b[3:]
                bindata[index] = b
        else:
            bindata = (bin(d)[2:] for d in __data)
    else:
        bindata = ("{:08b}".format(b) for b in struct.pack(">%s%s" % (len(__data), __dtype), *__data))

    bindata = "".join(bindata)
    binlen = len(bindata)

    if bitlength is None:
        return bindata

    elif bitlength == -1:
        if all(b == "0" for b in bindata):
            return "0"
        elif signed:
            return "0" + bindata.lstrip("0")
        return bindata.lstrip("0")

    elif binlen > bitlength:
        actual_binlen = len(bindata.lstrip("0"))
        if actual_binlen > bitlength:
            raise ValueError(f"data's bitlength({actual_binlen}) is longer than the given bitlength({bitlength})")
        bindata = bindata.removeprefix("0" * (binlen - bitlength))

    elif binlen < bitlength:
        bindata = bindata.zfill(bitlength)

    return bindata


def frombin(data: List[int], dtype: Optional[Intdtype] = None, num: int = 1) -> List[int] | int:
    if dtype is None:
        return int(data, 2)
    byte_data = int(data, 2).to_bytes((len(data) + 7) // 8, byteorder=ENDIAN)
    decoded_data = list(struct.unpack("%s%s%s" % (">" if ENDIAN == "big" else "<", num, dtype), byte_data))
    return decoded_data if num != 1 else decoded_data[0]
