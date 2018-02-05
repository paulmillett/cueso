/*
 * PFAppKernels.cpp
 * Copyright (C) 2018 Joseph Carmack <joseph.liping@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "PFAppKernels.h"
#include <stdio.h>



// -------------------------------------------------------
// Device Functions
// -------------------------------------------------------



/*********************************************************
  * Compute the laplacian of a field 'f' using periodic
  * boundary conditions (PBC).
  *******************************************************/

__device__ double laplacianPBC(double* f,int gid, int x, int y, int z, int nx, int ny, int nz, double h)
{
    // get id of neighbors with periodic boundary conditions
    int xlid,xrid,ylid,yrid,zlid,zrid;
    
    // x neighbor ids
    if(x == 0) xlid = nx*ny*z + nx*y + nx-1;
    else xlid = nx*ny*z + nx*y + x-1;
    if(x == nx-1) xrid = nx*ny*z + nx*y + 0;
    else xrid = nx*ny*z + nx*y + x+1;
    
    // y neighbor ids
    if(y == 0) ylid = nx*ny*z + nx*(ny-1) + x;
    else ylid = nx*ny*z + nx*(y-1) + x;
    if(y == ny-1) yrid = nx*ny*z + nx*0 + x;
    else yrid = nx*ny*z + nx*(y+1) + x;

    // z neighbor ids
    if(z == 0) zlid = nx*ny*(nz-1) + nx*y + x;
    else zlid = nx*ny*(z-1) + nx*y + x;
    if(z == nz-1) yrid = nx*ny*0 + nx*y + x;
    else zrid = nx*ny*(z+1) + nx*y + x;

    // get values of neighbors
    double xl = f[xlid];
    double xr = f[xrid];
    double yl = f[ylid];
    double yr = f[yrid];
    double zl = f[zlid];
    double zr = f[zrid];
    double lap = (xl+xr+yl+yr+zl+zr-6.0*f[gid])/(h*h);
    return lap;
}



/*********************************************************
  * Compute the chemical potential using the derivative
  * of the free energy density functional with respect
  * to the concentration c:
  *
  * free energy densisty = volInt( 
  *                                 c^4 - 2c^3 + c^2
  *                                 - 1/2*kap*|grad(c)|^2 
  *                              )
  *
  * df/dc = derivative of free energy density
  *       = (
  *         4c^3 - 6c^2 + 2c - kap*lap(c)
  *         )
  *******************************************************/

__device__ double zeroToOneDoubleWell(double c, double lap_c, double kap)
{
    return 4.0*c*c*c - 6.0*c*c + 2.0*c - kap*lap_c;
}



// -------------------------------------------------------
// Device Kernels for Testing
// -------------------------------------------------------




/*********************************************************
  * Kernel for unit testing the laplacianPBC device 
  *******************************************************/

__global__ void testLap(double* f, int nx, int ny, int nz, double h)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        f[gid] = laplacianPBC(f,gid,idx,idy,idz,nx,ny,nz,h);
    }
}



// -------------------------------------------------------
// Device Kernels for Simulation
// -------------------------------------------------------



/*********************************************************
  * Compute the laplacian of the concentration array c
  * and store it in the device array df.
  *******************************************************/

__global__ void calculateLap(double* c,double* df, int nx, int ny, int nz, double h)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        df[gid] = laplacianPBC(c,gid,idx,idy,idz,nx,ny,nz,h);
    }
}



/*********************************************************
  * Computes the chemical potential of a concentration
  * order parameter and stores it in the df_d array.
  *******************************************************/

__global__ void calculateChemPot(double* c,double* df, double kap, int nx, int ny, int nz)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        double cc = c[gid];
        double lap_c = df[gid];
        // compute the chemical potential
        df[gid] = zeroToOneDoubleWell(cc,lap_c,kap);
    }
}



/*********************************************************
  * Computes the chemical potential laplacian, multiplies
  * it by Mobility and time step to get the RHS of the CH
  * equation, then uses this RHS value to perform an
  * Euler update of the concentration in time.
  *******************************************************/

__global__ void lapChemPotAndUpdate(double* c,double* df,double M, double dt, int nx, int ny, int nz, double h)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        // compute chemical potential laplacain with PBC
        double lap = laplacianPBC(df,gid,idx,idy,idz,nx,ny,nz,h);
        // form CH eqn RHS and use to update c
        c[gid] += M*dt*lap;
    }
}



/*********************************************************
  * Copies the contents of c into cpyBuffer so the c data
  * can be asynchronously transfered from the device to
  * the host.
  *******************************************************/

__global__ void populateCopyBuffer(double* c,double* cpyBuff, int nx, int ny, int nz)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        // copy the contents of c to cpyBuff
        cpyBuff[gid] = c[gid];
    }
}
