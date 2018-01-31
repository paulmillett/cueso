/*
 * cudaErrorMacros.h
 * Copyright (C) 2018 Joseph Carmack <joseph.liping@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef CUDAERRORMACROS_H
#define CUDAERRORMACROS_H

// cuda error macro for asynchronous errors (i.e. kernel errors)
// only include these for debug builds
#ifdef DEBUG
#define cudaCheckAsyncErrors(msg) { \
    cudaError_t asyncErr = cudaDeviceSynchronize(); \
    if (asyncErr != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(cudaGetLastError()), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    }}
#else
#define cudaCheckAsyncErrors(msg) 
#endif

// cuda error macro for synchronous errors (i.e. allocating 
// memory & copying to/from device)
#define cudaCheckErrors(msg) { \
    cudaError_t syncErr = cudaGetLastError(); \
    if (syncErr != cudaSuccess) { \
        fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
            msg, cudaGetErrorString(syncErr), \
            __FILE__, __LINE__); \
        fprintf(stderr, "*** FAILED - ABORTING\n"); \
        exit(1); \
    }}

#endif /* !CUDAERRORMACROS_H */
