#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:07:47 2024

@author: paul
"""
import numpy as np
import pyFCS as fcs
# from tqdm import tqdm


log = list()

def tryblock(func, error, *args, name=str(), i=None, **kwargs):
    try:
        _ = func(*args, **kwargs)
    except error:
        pass
    except Exception as e:
        log.append((i, name, e))
    else:
        log.append(f"iteration {i}:/{name}/ did not produce the expected error of /{error}/")


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

def compute_edges(A, B, offset=2):
    if np.issubdtype(type(A[0]), np.number):
        return compute_edge(A,B, offset=offset)
    return np.array([compute_edge(a, b, offset=offset) for a, b in zip(A, B)])

def main():
    for i in range(100):
        data = np.array([np.cumsum(np.random.randint(1,300,size=np.random.randint(2,30))) for _ in range(20)], dtype=object)
        datau = np.array([d.astype(np.uint64) for d in data], dtype=object)
        dat = data[0]
        datu = datau[0]
        
        nanos = np.array([np.random.randint(0,100, size=d.size) for d in data], dtype=object)
        nanosu = np.array([n.astype(np.uint64) for n in nanos], dtype=object)
        na = nanos[0]
        nau = nanosu[0]
        
        weightsI = np.array([np.exp(-i*np.arange(max(n.max()+1 for n in nanos))) for i in range(1,3)])
        print(f"weightsI shape: {weightsI.shape}")
        weightsI_err = weightsI[:,:-1]
        weiI = weightsI[0,:]
        weiI_err = np.exp(-np.arange(na.max()))
        
        weights = np.empty(data.shape, dtype=object)
        for j in range(weights.size):
            weights[j] = np.random.normal(0.0, 3.0, size=(3, data[j].size))
        weights_s = np.array([w[0,:] for w in weights], dtype=object)
        wei = weights[0]
        wei_s = wei[0,:]
        
        
        
        weights_dimerr = np.empty(data.shape, dtype=object)
        for j in range(weights.size):
            weights_dimerr[j] = np.random.normal(0.0, 3.0, size=(2,2, data[j].size))
        weights_lenerr = np.empty(data.shape, dtype=object)
        for j in range(weights.size):
            weights_lenerr[j] = np.random.normal(0.0, 3.0, size=(2, data[j].size+3))
        wei_dimerr = weights_dimerr[0]
        wei_lenerr = weights_lenerr[0]
        
        edges = compute_edges(data, data, 2)
        edgesu = compute_edges(datau, datau, 2)
        
        edges_err = compute_edges(data, data, -2)
        edges_erru = compute_edges(datau, datau, -2)
        
        edg = edges[0]
        edg_err = edges_err[0]
        edgu = edgesu[0]
        edg_erru = edges_erru[0]
        edg_max_err = edges.min(axis=0)
        edg_min_err = edges.max(axis=0)
        edg_outer = np.array([edges[:,0].min(), edges[:,1].max()])
        edg_outer_err = np.array([edges[:,0].max()+1, edges[:,1].min()-1])
        
        bins = np.arange(20)
        binsu = bins.astype(np.uint64)
        print("basic correlate")
        out = fcs.correlate(data, datau, bins)
        ou = fcs.correlate(dat, datu, bins)
        outu = fcs.correlate(datau, datau, binsu)
        ouu = fcs.correlate(datu, datu, bins)
        aout = fcs.correlate(data, data, bins)
        aoutu = fcs.correlate(datau, datau, binsu)
        aouu = fcs.correlate(datu, datu, bins)
        print("with weights")
        awouu = fcs.correlate(datu, datu, bins, wei_s, wei_s)
        awou = fcs.correlate(dat, dat, binsu, wei, wei)
        awoutu = fcs.correlate(datau, datau, binsu, weights_s, weights_s)
        awout = fcs.correlate(data, datau, binsu, weights, weights)
        print("with nanos")
        Iout = fcs.correlate(data, datau, bins, weightsI, weightsI, nanos, nanosu)
        Iou = fcs.correlate(dat, datu, bins, weightsI, weightsI, nau, na)
        Ioutu = fcs.correlate(datau, datau, binsu, weightsI, weightsI, nanos, nanos)
        iout = fcs.correlate(data, datau, bins, weiI, weiI, nanos, nanosu)
        iou = fcs.correlate(dat, datu, bins, weiI, weiI, nau, na)
        ioutu = fcs.correlate(datau, datau, binsu, weiI, weiI, nanos, nanos)
        print("normalizations")
        norm = fcs.normalization_factor(data, data, bins)
        normu = fcs.normalization_factor(datau, datau, binsu)
        nor = fcs.normalization_factor(dat, dat, bins)
        noru = fcs.normalization_factor(datu, datu, binsu)
        print("changing norm")
        norm += 1.0
        print('changing normu')
        normu += 1.0
        print("changing nor")
        nor += 2.0
        print("changing noru")
        noru += 2.3
        print("another norm calculation")
        norm = fcs.normalize(norm, data, data, bins)
        print("another norm calculation")
        normu = fcs.normalize(norm.astype(np.int64), datau, datau, binsu)
        print("another norm calculation")
        nor = fcs.normalize(nor, dat, dat, bins)
        print("another norm calculation")
        noru = fcs.normalize(nor.astype(np.int64), datu, datu, binsu)
        print("another changin ov norm")
        norm += 1.0
        print("another chaing normu")
        normu += 1.0
        print('another changing nor')
        nor += 2.0
        print('another changing noru')
        noru += 2.3
    
        print('second round')
        out = fcs.correlate(data, datau, bins, edges=edges)
        ou = fcs.correlate(dat, datu, bins, edges=edgu)
        outu = fcs.correlate(datau, datau, binsu, edges=edgesu)
        ouu = fcs.correlate(datu, datu, bins, edges=edg)
        aout = fcs.correlate(data, data, bins, edges=edgesu)
        aoutu = fcs.correlate(datau, datau, binsu, edges=edges)
        aouu = fcs.correlate(datu, datu, bins, edges=edgu)
        aout = fcs.correlate(data, data, bins, edges=edg_outer)
        awouu = fcs.correlate(datu, datu, bins, wei_s, wei_s)
        awou = fcs.correlate(dat, dat, binsu, wei, wei)
        awoutu = fcs.correlate(datau, datau, binsu, weights_s, weights_s)
        awout = fcs.correlate(data, datau, binsu, weights, weights)
        Iout = fcs.correlate(data, datau, bins, weightsI, weightsI, nanos, nanosu)
        Iou = fcs.correlate(dat, datu, bins, weightsI, weightsI, nau, na)
        Ioutu = fcs.correlate(datau, datau, binsu, weightsI, weightsI, nanos, nanos)
        iout = fcs.correlate(data, datau, bins, weiI, weiI, nanos, nanosu)
        iou = fcs.correlate(dat, datu, bins, weiI, weiI, nau, na)
        ioutu = fcs.correlate(datau, datau, binsu, weiI, weiI, nanos, nanos)
        # print("tryblocks")
        tryblock(fcs.correlate, TypeError, "hello", datu, bins, name="bad timesT", i=i)
        tryblock(fcs.correlate, ValueError, dat, dat, bins[::-1], name="non monotonic bins", i=i)
        tryblock(fcs.correlate, ValueError, dat, data, bins, name="mismatch # arrays", i=i)
        tryblock(fcs.correlate, ValueError, dat[::-1], dat, bins, name="reversed times", i=i)
        tryblock(fcs.correlate, ValueError, np.array([d[::-1] if j == 1 else d for j, d in enumerate(data)], dtype=object), 
                 data, bins, name="mismatch # arrays", i=i)
        tryblock(fcs.correlate, ValueError, data, datau, bins, edges=edges_err, name="edges err, type edges_err", i=i)
        tryblock(fcs.correlate, ValueError, data, datau, bins, edges=edges_erru, name="edges err, type edge_erru", i=i)
        tryblock(fcs.correlate, ValueError, dat, datu, bins, edges=edg_err, name="edges err, type edg_err", i=i)
        tryblock(fcs.correlate, ValueError, dat, datu, bins, edges=edg_erru, name="edges err, type edg_erru", i=i)
        tryblock(fcs.correlate, ValueError, data, datau, bins, edges=edg_min_err, name="edges err, type edg_min_err", i=i)
        tryblock(fcs.correlate, ValueError, data, datau, bins, edges=edg_max_err, name="edges err, type edg_max_err", i=i)
        tryblock(fcs.correlate, ValueError, data, datau, bins, edges=edg_outer_err, name="edges err, type edg_outer_err", i=i)
        tryblock(fcs.correlate, TypeError, dat, dat, bins, weights_s, weights_s, name="weights wrong number of arrays", i=i)
        tryblock(fcs.correlate, TypeError, dat, dat, bins, wei_s, weights_s, name="weights wrong number of arrays, mixed", i=i)
        tryblock(fcs.correlate, ValueError, data, data, bins, wei_s, wei_s, name="weights wrong number of arrays, too few", i=i)
        tryblock(fcs.correlate, ValueError, data, data, bins, weights_s, weights, name="weights mixed number of weights", i=i)
        tryblock(fcs.correlate, TypeError, data, data, bins, weights_dimerr, weights_dimerr, name="weights too many dimensions", i=i)
        tryblock(fcs.correlate, ValueError, data, data, bins, weights_lenerr, weights_lenerr, name="weights too many dimensions", i=i)
        tryblock(fcs.correlate, ValueError, dat, dat, bins, wei_lenerr, wei_lenerr, name="weights too many dimensions, single", i=i)
        tryblock(fcs.correlate, TypeError, dat, dat, bins, wei_dimerr, wei_dimerr, name="weights too many dimensions, single", i=i)
        tryblock(fcs.correlate, ValueError, dat, dat, bins, weiI_err, weiI, na, na, name="bad weiI_err  1", i=i)
        tryblock(fcs.correlate, ValueError, data, data, bins, weightsI_err, weightsI, nanos, nanos, name="bad weightsI_err  2", i=i)
        tryblock(fcs.correlate, ValueError, dat, dat, bins, weiI, weiI_err, na, na, name="bad weiI_err  3", i=i)
        tryblock(fcs.correlate, ValueError, data, data, bins, weightsI, weightsI_err, nanos, nanos, name="bad weightsI_err  4", i=i)
        tryblock(fcs.correlate, ValueError, data, data, bins, weiI, weightsI, nanos, nanos, name="inconsistent nW  5", i=i)
        
        dat += 10
        datu += 10
        bins += 10
        binsu += 10
        out += 10
        ou += 10
        outu += 10
        ouu += 10
        aout += 10
        aoutu += 10
        aouu += 10
        awouu += 3.2
        awou += -3.2
        awoutu += 5.7
        awout += 6.8
        Iout += 1.0
        Iou += 1.0
        Ioutu += 8.1
        iout += 7.4
        iou += 1.52
        ioutu += 62

        edges = compute_edges(data, data, 2)
        edges_err = compute_edges(data, data, -2)
        edgesu = compute_edges(datau, datau, 2)
        edges_erru = compute_edges(datau, datau, -2)
        edg = edges[0]
        edgu = edgesu[0]
        print("third round")
        out = fcs.correlate(data, datau, bins)
        ou = fcs.correlate(dat, datu, bins)
        outu = fcs.correlate(datau, datau, binsu)
        ouu = fcs.correlate(datu, datu, bins)
        aout = fcs.correlate(data, data, bins)
        aoutu = fcs.correlate(datau, datau, binsu)
        aouu = fcs.correlate(datu, datu, bins)
        out = fcs.correlate(data, datau, bins, edges=edges)
        ou = fcs.correlate(dat, datu, bins, edges=edgu)
        outu = fcs.correlate(datau, datau, binsu, edges=edgesu)
        ouu = fcs.correlate(datu, datu, bins, edges=edg)
        aout = fcs.correlate(data, data, bins, edges=edgesu)
        aoutu = fcs.correlate(datau, datau, binsu, edges=edges)
        aouu = fcs.correlate(datu, datu, bins, edges=edgu)
        awouu = fcs.correlate(datu, datu, bins, wei_s, wei_s)
        awou = fcs.correlate(dat, dat, binsu, wei, wei)
        awoutu = fcs.correlate(datau, datau, binsu, weights_s, weights_s)
        awout = fcs.correlate(data, datau, binsu, weights, weights)
        Iout = fcs.correlate(data, datau, bins, weightsI, weightsI, nanos, nanosu)
        Iou = fcs.correlate(dat, datu, bins, weightsI, weightsI, nau, na)
        Ioutu = fcs.correlate(datau, datau, binsu, weightsI, weightsI, nanos, nanos)
        iout = fcs.correlate(data, datau, bins, weiI, weiI, nanos, nanosu)
        iou = fcs.correlate(dat, datu, bins, weiI, weiI, nau, na)
        ioutu = fcs.correlate(datau, datau, binsu, weiI, weiI, nanos, nanos)
        log = list()
    
if __name__ == "__main__":
    main()
    if not log:
        print("no errors reported")
    elif all(isinstance(l, str) for l in log):
        for l in log:
            print(l)
        print('all no errors')
    elif all(isinstance(l, tuple) for l in log):
        for l in log:
            print(l)
        print("all different error type")
