import struct
from typing import List, Iterable, Literal, Optional

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


def frombin(
    __data: Bitcode,
    __dtype: Intdtype,
    num: int = 1,
    *,
    encoding: Optional[str] = None,
    signed: bool = True,
) -> List[int] | int:
    if __dtype is int:
        stop = len(__data)
        step = stop // num
        if signed:
            decoded_data = [None] * num
            for index, i in enumerate(range(0, stop, step)):
                bindata = __data[i : i + step]
                decoded_data[index] = int("-%s" % (bindata) if bindata[0] == "1" else bindata, 2)
        else:
            decoded_data = [int(__data[i : i + step], 2) for i in range(0, stop, step)]
        return decoded_data if num != 1 else decoded_data[0]

    bytedata = int(__data, 2).to_bytes((len(__data) + 7) // 8, ENDIAN)
    try:
        decoded_data = list(struct.unpack("%s%s%s" % (">" if ENDIAN == "big" else "<", num, __dtype), bytedata))
        return decoded_data if num != 1 else decoded_data[0]
    except struct.error:
        raise TypeError(f"cannot convert byte data to '{__dtype}'")


def tobytes(
    __data: int,
    __dtype: Intdtype,
    bytelength: Optional[int] = None,
    *,
    signed: bool = True,
) -> bytes:
    def check_bytelength(bytedata: bytes) -> bytes:
        encoded_bytelen = len(bytedata)

        if bytelength == -1:
            if signed:
                return bytes(1) + bytedata.lstrip(bytes(1))
            return bytedata.lstrip(bytes(1))

        elif encoded_bytelen > bytelength:
            actual_binlen = len(bytedata.lstrip(bytes(1)))
            if actual_binlen > bytelength:
                raise ValueError(
                    f"data's bytelength({actual_binlen}) is longer than the given bytelength({bytelength})"
                )
            bytedata = bytedata.removeprefix(bytes(1) * (encoded_bytelen - bytelength))

        elif encoded_bytelen < bytelength:
            bytedata = bytes(bytelength - encoded_bytelen) + bytedata

        return bytedata

    if __dtype == "bin":
        bytedata = int(__data, 2).to_bytes((len(__data) + 7) // 8, ENDIAN)
        return bytedata if bytelength is None else check_bytelength(bytedata)

    elif __dtype == int:
        bytedata = int.to_bytes(__data, (__data.bit_length() + 7) // 8, ENDIAN, signed=signed)
        return bytedata if bytelength is None else check_bytelength(bytedata)

    if not isinstance(__data, Iterable):
        __data = [__data]

    bytedata = [None] * len(__data)
    if bytelength is not None:
        for i, d in enumerate(__data):
            converted_data = struct.pack("%s%s" % (">" if ENDIAN == "big" else "<", __dtype), d)
            bytedata[i] = check_bytelength(converted_data)
    else:
        for i, d in enumerate(__data):
            converted_data = struct.pack("%s%s" % (">" if ENDIAN == "big" else "<", __dtype), d)
            bytedata[i] = converted_data

    return bytedata


def frombytes(
    __data: bytes,
    __dtype: Intdtype,
    num: int = 1,
    *,
    encoding: Optional[str] = None,
    signed: bool = True,
) -> List[int] | int:
    """converts a string of 0 and 1 back into the original data

    Args:
        __data (BinaryCode): a string of 0 and 1
        __dtype (Union[int, float, str]): the desired data type to convert to

    Raises:
        TypeError: if the desired datatype is not of the integer, floats or strings data type

    Returns:
        Union[int, float, str]: converted data
    """

    if __dtype is int:
        stop = len(__data)
        step = stop // num
        decoded_data = [int.from_bytes(__data[i : i + step], ENDIAN, signed=signed) for i in range(0, stop, step)]
        return decoded_data if num != 1 else decoded_data[0]

    if __dtype == "bin":
        return "".join("{:08b}".format(b) for b in __data)

    else:
        try:
            decoded_data = list(struct.unpack("%s%s%s" % (">" if ENDIAN == "big" else "<", num, __dtype), __data))
            return decoded_data if num != 1 else decoded_data[0]
        except struct.error:
            raise TypeError(f"cannot convert byte data to '{__dtype}'")
