/*
 * PFAppKernels.h
 * Copyright (C) 2018 joseph Carmack <joseph.liping@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef PFAPPKERNELS_H
#define PFAPPKERNELS_H

// kernel for evolving c-field using finite difference to solve
// the Cahn-Hilliard equation
__global__ void calculateLap(double* c,double* df, int nx, int ny, int nz,double h);


__global__ void calculateChemPot(double* c,double* df, double kap, int nx, int ny, int nz);


__global__ void lapChemPotAndUpdate(double* c,double* df,double M,double dt, int nx, int ny, int nz,double h);


__global__ void populateCopyBuffer(double* c,double* cpyBuff, int nx, int ny, int nz);


// kernel for testing the laplacian function
__global__ void testLap(double* f, int nx, int ny, int nz, double h);

#endif /* !PFAPPKERNELS_H */
