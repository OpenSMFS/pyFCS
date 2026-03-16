import numpy as np
import pyFCS as fcs
from itertools import product, permutations
import warnings

bns_u = np.logspace(0,10, 10).astype(np.uint64)
bns_i = bns_u.astype(np.int64)
bns_li = list(bns_i)
bns_lu = list(bns_u)
bns_l = list(int(i) for i in bns_i)
bns_ti = tuple(bns_li)
bns_tu = tuple(bns_lu)
bns_t = tuple(bns_l)
bins_c = (bns_u, bns_i, bns_li, bns_lu, bns_l, bns_t, bns_ti, bns_tu)

def gen_time(center:float, sigma:float, lam:int)->np.ndarray:
    """
    generate a single array of times
    """
    return np.sort(np.random.normal(center, sigma, size=np.random.poisson(lam))).astype(np.int64)


def gen_times(n:int, dur:int, sigma:float, lam:int)->tuple[np.ndarray,np.ndarray]:
    """
    generate a set of times, each a sequence of arrays (returned as object)
    """
    centers = np.sort(np.random.randint(int(sigma*2), dur, size=n))
    tmsA = np.array([gen_time(c, sigma, lam) for c in centers], dtype=object)
    tmsB = np.array([gen_time(c, sigma, lam) for c in centers], dtype=object)
    return tmsA, tmsB

# tuple of possible kwarg combinations
bool_combs = (True, False, None, np.array([1,2,3]), 'None')

kwargs = tuple({n:v for n, v in zip(("normalize", "norm_bin_width", "minzero"), vals) if v != 'None'} 
               for vals in product(bool_combs, bool_combs, bool_combs))

def compute_edge(A, B, offset=2):
    """Compute, for a single pair of input arrays, the edges, with offset to
    add to edge to shift the start/stop"""
    if isinstance(offset, str) and offset == 'rand':
        offset = np.random.randint(0, 100, size=2)
        offset[0] = -offset[0]
    elif np.issubdtype(type(offset), np.integer):
        offset = np.array([-offset, offset], dtype=np.int64)
    edg = np.array([min(np.min(A), np.min(B)), max(np.max(A), np.max(B))], dtype=np.int64)
    edg += offset
    edg[edg<0] = 0
    return edg

ismutable = lambda obj : isinstance(obj, (np.ndarray, list))
isflat = lambda obj : np.issubdtype(type(obj[0]), np.number)

def compute_edges(A, B, offset=2):
    if np.issubdtype(type(A[0]), np.number):
        return compute_edge(A,B, offset=offset)
    return np.array([compute_edge(a, b, offset=offset) for a, b in zip(A, B)])

def convert_to(x, cast_func, outer_type):
    return outer_type([cast_func(a) for a in x])

def as_uint64(x): return x.astype(np.uint64)
def as_int64(x): return x.astype(np.int64)
def as_int(x): return [int(i) for i in x]

cast_funcs = (as_uint64, as_int64, as_int)

def as_npobject(x): return np.array([a for a in x], dtype=object)
def as_list(x): return list(x)
def as_tuple(x): return tuple(x)
def as_flat_array(x): return np.sort(np.concatenate(x))
def as_flat_list(x): return list(np.sort(np.concatenate(x)))
def as_flat_tuple(x): return list(np.sort(np.concatenate(x)))

outer_funcs = (as_npobject, as_flat_array, as_list, as_flat_list, as_tuple, as_flat_tuple)


# preallocating tuples of various combinations to test
def cororders(): return permutations(range(2), 2)
def kworders(): return permutations(range(len(kwargs)), len(kwargs))
def castfuncs(): return product(cast_funcs, outer_funcs)

        
def inplace_add(inp, n):
    """special function that performs inplace add to anything except tuple"""
    if isflat(inp):
        if isinstance(inp, np.ndarray):
            inp += n
        elif isinstance(inp, list):
            for i in range(len(inp)):
                inp[i] += n
    else:
        for sub in inp:
            inplace_add(sub, n)


timesErrors = (np.array([1.2,1.5,4.5], dtype=np.float64), np.array([[1,3,4,5,7], [1,3,4,5,7]], dtype=np.int64), np.array([], dtype=np.uint64),
               [np.array([1,3,5,6]), "hello world"], ("hello world", np.array([4,10,44])))

binsErrors = ("hello", np.array([1.2, 2.3,5.0]), np.array([10,1,20], dtype=np.int64), np.array([1], dtype=np.uint64))

def ordercombs(): return product(bins_c, castfuncs(), castfuncs(), cororders())

prevlist = list()
# begin running the tests in giant loop
#repeat whole process 10 times
# for i in range(10):
#     timesO = gen_times(20, 10000000, 4000.0, 40)
#     ii = 0
#     for bins, (cfuncA, ofuncA), (cfuncB, ofuncB), (A, B) in ordercombs():
#         ii += 1
#         # convert times to the desired type
#         timesA = convert_to(timesO[A], cfuncA, ofuncA)
#         timesB = convert_to(timesO[B], cfuncB, ofuncB)
#         edges = compute_edges(timesA, timesB, offset=2)
#         for kw in kwargs:
#             curlist = list()
#             if any(isinstance(item, np.ndarray) for item in kw.items()) or ((isflat(timesA) or (len(timesA) == 1)) != (isflat(timesA) or (len(timesA) == 1))):
#                 try:
#                     temp = fcs.correlate(timesA, timesB, bins, **kw)
#                 except ValueError:
#                     pass
#                 except Exception as e:
#                     print(f"Unexpected exception type raised: {e}")
#                 else:
#                     print(f"exception not raise for {kw}, when exception expected")
#             else:
#                 temp = fcs.correlate(timesA, timesB, bins, **kw)
#                 curlist.append(temp)
#                 temp += 12
#                 inplace_add(timesA, 4)
#                 inplace_add(timesB, 4)
#                 if not kw.get('minzero', False):
#                     temp = fcs.correlate(timesA, timesB, bins, edges=edges, **kw)
#                     curlist.append(temp)
#                     temp += 2
#                 else:
#                     with warnings.catch_warnings(record=True) as w:
#                         temp = fcs.correlate(timesA, timesB, bins, edges=edges, **kw)
#                         if len(w) == 0:
#                             print("correlating with edges and minzero didn't raise a warning")
#                         elif len(w) > 1:
#                             print("correlate raise multiple unexpected warings", w)
#                         elif not issubclass(w[0].category, UserWarning):
#                             print(f"correlate raised an unexpected type of warning: {w[0].category}")
#                 inplace_add(timesA, -4)
#                 inplace_add(timesB, -4)
#                 curlist.append(temp)
#                 for cur in curlist:
#                     cur -= 2
#                 for prev in prevlist:
#                     prev -= 4
#                 prevlist = curlist