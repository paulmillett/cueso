/*
 * PFSIPSKernels.h
 * 
 *
 * Distributed under terms of the MIT license.
 */

#ifndef PFSIPSKERNELS_H
#define PFSIPSKERNELS_H

// kernel for evolving c-field using finite difference to solve
// the Cahn-Hilliard equation

__global__ void calculateLapBoundaries(double* c, double* df, int nx, int ny, int nz, 
													double h, bool bX, bool bY, bool bZ);


__global__ void calculateChemPotFH(double* c, double* df, double* chi, double kap, double A, 
											  double chiCond, double chiPS, double chiPN, double N, 
											  int nx, int ny, int nz, int current_step, double dt);


__global__ void calculateMobility(double* c,double* Mob,double M,int nx,int ny,int nz,
											 double phiCutoff, double N,
        									 double gamma, double nu, double D0, double Mweight, double Mvolume);


__global__ void lapChemPotAndUpdateBoundaries(double* c,double* df,double* Mob, double M, double dt, 
															 int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ);


__global__ void populateCopyBufferSIPS(double* c, double* cpyBuff, int nx, int ny, int nz);


__global__ void populateCopyBufferMOB(double* Mob, double* cpyBuff, int nx, int ny, int nz);


// kernel for testing the laplacian function
__global__ void testLapSIPS(double* f, int nx, int ny, int nz, double h, 
									 bool bX, bool bY, bool bZ);

#endif /* !PFSIPSKERNELS_H */