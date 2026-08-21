"""
Author: Jack Cochran
File: Implementation of bit accurate python archtitectural model of FFT for RTL Development

"""

from __future__ import annotations
import numpy as np

def numpy_fft(input_data):
    return np.fft.fft(input_data)

"""
Custom fft written by me
Q1.15 fixed point format
"""
def custom_fft(input_data):
    reordered_input = apply_bitrev(input_data)
    return reordered_input
    

"""
Address bit reverse method for array of 512 fxp Q1.15 inputs 
"""
def apply_bitrev(input_fxp_arr, p):
    if isinstance(input_fxp_arr, np.ndarray):
        return input_fxp_arr[p]
    # return [input_fxp_arr[int(j)] for j in p]



"""
Permutation table generation
"""
def bitrev_table(n: int) -> np.ndarray:
    w = _width(n)
    p = np.zeros(1, dtype=np.int32)
    for _ in range(w):
        p = np.concatenate([p * 2, p * 2 + 1])
    return p



 
"""
Twiddle factor array generation
"""
def gen_twiddle_arr(n: int, w_tw: int = 16) -> list[tuple[int, int]]:
    scale = (1 << (w_tw - 1)) - 1
    lo, hi = -(1 << (w_tw - 1)), scale
    out = []
    for w in gen_twiddle_float(n):
        re = int(np.rint(w.real * scale))
        im = int(np.rint(w.imag * scale))
        out.append((max(lo, min(hi, re)), max(lo, min(hi, im))))
    return out

def gen_twiddle_float(n: int) -> np.ndarray:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"N must be a positive power of two, got {n}")
    t = np.arange(n // 2)
    return np.exp(-2j * np.pi * t / n)


def _width(n: int) -> int:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"N must be a positive power of two, got {n}")
    return n.bit_length() - 1


def bitrev_table_oracle(n: int) -> list[int]:
    w = _width(n)
    return [int(f"{i:0{w}b}"[::-1], 2) for i in range(n)]





