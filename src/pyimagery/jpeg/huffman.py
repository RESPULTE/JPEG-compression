from heapq import heapify, heappop, heappush
from collections import Counter
from typing import Literal, Tuple, Dict, List

from pyimagery.utils.binary import frombin, tobin
from pyimagery._type_hint import Bitcode


class DecodingError(Exception):
    pass


class EncodingError(Exception):
    pass


class CorruptedHeaderError(Exception):
    pass


class CorruptedEncodingError(Exception):
    pass


def decode(codebook: Dict[int, Bitcode], encoded_data: Bitcode) -> List[int]:
    decoded_data = [None] * len(encoded_data)

    to_process = encoded_data
    curr_index = 0
    curr_code = ""

    while to_process:
        curr_code += to_process[:1]
        to_process = to_process[1:]

        if curr_code not in codebook:
            continue

        curr_elem_binsize = codebook[curr_code]
        curr_elem = frombin(to_process[:curr_elem_binsize], int)
        to_process = to_process[curr_elem_binsize:]

        decoded_data[curr_index] = curr_elem
        curr_index += 1
        curr_code = ""

    decoded_data = decoded_data[:curr_index]

    return decoded_data


# * done
def encode(dataset: List[int], dtype: Literal["AC", "DC"]) -> Tuple[Dict[int, Bitcode], Bitcode]:
    if dtype == "AC":
        dataset_size = len(dataset)
        bindataset = [None] * dataset_size
        prefix_to_encode = [None] * dataset_size

        for index, (zero_runlength, data) in enumerate(dataset):
            bin_data = tobin(data, int)
            bindataset[index] = bin_data
            prefix_to_encode[index] = (zero_runlength, len(bin_data))

    else:
        bindataset = [tobin(data, int) for data in dataset]
        prefix_to_encode = [len(data) for data in bindataset]

    codebook = generate_canonical_codebook(prefix_to_encode)
    encoded_data = "".join(
        x for bindata, binprefix in zip(bindataset, prefix_to_encode) for x in (codebook[binprefix], bindata)
    )

    return codebook, encoded_data


# TODO
def generate_header_from_codebook(codebook: Dict[int, Bitcode], dtype: Literal["AC", "DC"]) -> Bitcode:
    counted_codelengths = Counter([len(code) for code in codebook.values()])
    codelengths = ["0" * 8] * 16
    for length, count in counted_codelengths.items():
        codelengths[length - 1] = tobin(count, "B", bitlength=8)
    codelengths = "".join(codelengths)

    if dtype == "DC":
        symbols = [tobin(bitsize - 1, "B", bitlength=4) for bitsize in codebook.keys()]
    else:
        symbols = [
            ac_pair
            for zero_rle, bitsize in codebook.keys()
            for ac_pair in (tobin(zero_rle, "B", bitlength=4), tobin(bitsize - 1, "B", bitlength=4))
        ]

    return codelengths + "".join(symbols)


# * done
def generate_canonical_codebook(dataset: List[int]) -> Dict[int, Bitcode]:
    counted_dataset = Counter(dataset).most_common()
    to_process = [(count, 1, [symbol]) for symbol, count in counted_dataset]
    heapify(to_process)
    codebook = {symbol: 0 for symbol, _ in counted_dataset}

    while len(to_process) != 1:
        tree_freq_1, tree_max_bitlength_1, tree_1 = heappop(to_process)
        tree_freq_2, tree_max_bitlength_2, tree_2 = heappop(to_process)

        new_subtree = tree_1 + tree_2
        new_subtree_freq = tree_freq_1 + tree_freq_2
        new_subtree_max_bitlength = max(tree_max_bitlength_1, tree_max_bitlength_2) + 1

        for sym in new_subtree:
            codebook[sym] += 1

        balance_factor = 0
        if to_process:
            balance_factor = to_process[0][0]

        heappush(
            to_process,
            (
                new_subtree_freq + balance_factor,
                new_subtree_max_bitlength,
                new_subtree,
            ),
        )

    # just to ensure that the very first value will be zero
    curr_code = -1
    # making sure that the bit shift won't ever happen for the first value
    prev_bitlength = float("inf")
    # sort the codebook by the bitlength
    to_process = sorted([(bitlength, symbol) for symbol, bitlength in codebook.items()])

    canonical_codebook = {}
    for bitlength, symbol in to_process:

        # increment the code, which is in integer form btw, by 1
        # if the bitlength of this symbol is more than the last symbol, left-shift the code using bitwise operation
        curr_code += 1
        if bitlength > prev_bitlength:
            curr_code = curr_code << (bitlength - prev_bitlength)

        canonical_codebook[symbol] = tobin(curr_code, "I", bitlength=bitlength)
        prev_bitlength = bitlength

    return canonical_codebook


# TODO
def generate_codebook_from_header(header: Bitcode) -> Dict[Bitcode, int]:
    dtype, header = header[:16], header[16:]
    try:
        codelength_info = 8 * 16
        bin_codelengths, bin_symbols = header[:codelength_info], header[codelength_info:]

        num_symbols_per_codelength = [
            int(bin_codelengths[bitlen : bitlen + 8], 2) for bitlen in range(0, len(bin_codelengths), 8)
        ]

        num_codelength = len(num_symbols_per_codelength)
        if num_codelength != 16:
            raise ValueError(f"number of symbols decoded({num_codelength}) does not match the default values({16})")

        total_codes = sum(num_symbols_per_codelength)
        if dtype == "DC":
            symbols = frombin(bin_symbols, num=total_codes)
        else:
            symbols = [None] * total_codes
            bin_symbols = [bin_symbols[bitlen : bitlen + 8] for bitlen in range(0, len(bin_symbols), 8)]
            for index, bin_sym in enumerate(bin_symbols):
                symbols[index] = (frombin(bin_sym[:4]), frombin(bin_sym[4:]) + 1)

    except (IndexError, ValueError) as err:
        raise CorruptedHeaderError("Header cannot be decoded") from err

    codebook = {}
    curr_code = 0
    curr_sym_index = 0
    for bitlength, num in enumerate(num_symbols_per_codelength, start=1):

        for _ in range(num):
            bincode = tobin(curr_code, "H", bitlength=bitlength)
            codebook[bincode] = symbols[curr_sym_index]
            curr_sym_index += 1
            curr_code += 1

        curr_code = curr_code << 1

    return codebook
