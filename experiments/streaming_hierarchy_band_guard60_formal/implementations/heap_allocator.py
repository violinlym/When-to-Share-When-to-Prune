"""ctypes bridge; all persistent buffers contain scalars/indices, never KV vectors."""
import ctypes
from pathlib import Path
import numpy as np

_LIB = None


def allocate(importance,costs,order,stage,alive,n,lh,budget,allow_prune=True,examples=0):
    global _LIB
    if _LIB is None:
        _LIB=ctypes.CDLL(str(Path(__file__).with_name('libexp7_heap.so')))
        _LIB.allocate_heap.restype=ctypes.c_int64
        _LIB.allocate_heap.argtypes=[ctypes.c_int64]*4+[ctypes.c_void_p]*5+[ctypes.c_int]+[ctypes.c_void_p]*4+[ctypes.c_int64]
    importance=np.ascontiguousarray(importance,dtype=np.float32)
    costs=np.ascontiguousarray(costs,dtype=np.float32)
    order=np.ascontiguousarray(order,dtype=np.uint8)
    assert stage.dtype==alive.dtype==np.uint8 and stage.flags.c_contiguous and alive.flags.c_contiguous
    m=n*lh
    assert importance.shape==(m,) and costs.shape==order.shape==(m,2,3) and stage.shape==(m,2) and alive.shape==(m,)
    counters=np.zeros(16,np.uint64);by_head=np.zeros((lh,3),np.uint64)
    timings=np.zeros(2,np.float64);trace=np.zeros((examples,8),np.float64)
    pointer=lambda x:ctypes.c_void_p(x.ctypes.data)
    remaining=_LIB.allocate_heap(m,lh,n,budget,*map(pointer,[importance,costs,order,stage,alive]),int(allow_prune),
                                 *map(pointer,[counters,by_head,timings,trace]),examples)
    assert remaining>=0, ('heap allocator failure',remaining)
    return dict(remaining=remaining,counters=counters,by_head=by_head,scoring_seconds=float(timings[0]),
                heap_seconds=float(timings[1]),examples=trace[:int(counters[15])].tolist())


def masks(order,stage,alive,n,l,h):
    """Birth-major CPU state -> layer/KV/head/token/native-loop mask."""
    keep=np.ones((n*l*h,2,4),dtype=bool)
    for j in range(3):
        rows,components=np.nonzero(stage>j)
        keep[rows,components,order[rows,components,j]]=False
    keep[~alive.astype(bool)]=False
    return keep.reshape(n,l,h,2,4).transpose(1,3,2,0,4).copy()
