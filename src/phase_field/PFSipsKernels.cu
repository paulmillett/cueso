 /*
 * PFSipsKernels.cpp
 * Copyright (C) 2018 Joseph Carmack <joseph.liping@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#include "PFSipsKernels.h"
#include <stdio.h>
#include <math.h>
#include <curand.h>
#include <curand_kernel.h>



// -------------------------------------------------------
// Device Functions
// -------------------------------------------------------


/**********************************************************
   * Laplacian of non-uniform mobility field 
   * for Cahn-Hilliard Euler update
   ********************************************************/

__device__ double laplacianNonUniformMob(double *f, double *Mob,int gid, int x, int y, int z,
                                         int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ)
{
	// get id of neighbors for no-flux and PBCs
   int xlid,xrid,ylid,yrid,zlid,zrid;
   // -----------------------------------
   // X-Direction Boundaries
   // -----------------------------------
	if (bX) {
		// PBCs (x-dir.)
		if(x == 0) xlid = nx*ny*z + nx*y + nx-1;
		else xlid = nx*ny*z + nx*y + x-1;
		if(x == nx-1) xrid = nx*ny*z + nx*y + 0;
		else xrid = nx*ny*z + nx*y + x+1;
	}
	else {
	 	// no-flux BC (x-dir.)
		if (x == 0) xlid = nx*ny*z + nx*y + x;
		else xlid = nx*ny*z + nx*y + x-1;
		if (x == nx-1) xrid = nx*ny*z + nx*y + x;
		else xrid = nx*ny*z + nx*y + x+1;
   }
   // -----------------------------------
   // Y-Direction Boundaries
   // -----------------------------------
	if (bY) {
		// PBC Apply
	   if(y == 0) ylid = nx*ny*z + nx*(ny-1) + x;
    	else ylid = nx*ny*z + nx*(y-1) + x;
    	if(y == ny-1) yrid = nx*ny*z + nx*0 + x;
    	else yrid = nx*ny*z + nx*(y+1) + x;
   }
   else {
   	// no-flux BC (y-dir.)
      if(y == 0) ylid = nx*ny*z + nx*y + x;
    	else ylid = nx*ny*z + nx*(y-1) + x;
    	if(y == ny-1) yrid = nx*ny*z + nx*y + x;
    	else yrid = nx*ny*z + nx*(y+1) + x;
	}
   // -----------------------------------
   // Z-Direction Boundaries
   // -----------------------------------
	if (bZ) {
		// PBC Apply (z-dir.)
   	if(z == 0) zlid = nx*ny*(nz-1) + nx*y + x;
    	else zlid = nx*ny*(z-1) + nx*y + x;
    	if(z == nz-1) zrid = nx*ny*0 + nx*y + x;
    	else zrid = nx*ny*(z+1) + nx*y + x;
   }
	else {
		// no-flux BC (z-dir.)
		if(z == 0) zlid = nx*ny*z + nx*y + x;
    	else zlid = nx*ny*(z-1) + nx*y + x;
    	if(z == nz-1) zrid = nx*ny*z + nx*y + x;
    	else zrid = nx*ny*(z+1) + nx*y + x;
	}

    // ------------------------------------------
    // begin laplacian
	// ------------------------------------------
	
    // get values of neighbors for mobility
    double mobXl = Mob[xlid];
    double mobXr = Mob[xrid];
    double mobYl = Mob[ylid];
    double mobYr = Mob[yrid];
    double mobZl = Mob[zlid];
    double mobZr = Mob[zrid];
    // get values of neighbors for mu
    double xl = f[xlid];
    double xr = f[xrid];
    double yl = f[ylid];
    double yr = f[yrid];
    double zl = f[zlid];
    double zr = f[zrid];
    // get value of current points
    double bo = Mob[gid];
    double fo = f[gid];
    // begin laplacian
    double bx1 = 2.0/(1.0/mobXl + 1.0/bo);
    double bx2 = 2.0/(1.0/mobXr + 1.0/bo);
    double by1 = 2.0/(1.0/mobYl + 1.0/bo);
    double by2 = 2.0/(1.0/mobYr + 1.0/bo);
    double bz1 = 2.0/(1.0/mobZl + 1.0/bo);
    double bz2 = 2.0/(1.0/mobZr + 1.0/bo);
    double lapx = (xl*bx1 + xr*bx2 - (bx1+bx2)*fo)/(h*h); 
    double lapy = (yl*by1 + yr*by2 - (by1+by2)*fo)/(h*h);
    double lapz = (zl*bz1 + zr*bz2 - (bz1+bz2)*fo)/(h*h);
    double lapNonUniform = lapx + lapy + lapz;
    return lapNonUniform;
}   
   

/*********************************************************
   * Compute Laplacian with user specified 
   * boundary conditions (UpdateBoundaries)
   ******************************************************/
	
__device__ double laplacianUpdateBoundaries(double* f,int gid, int x, int y, int z, 
								            int nx, int ny, int nz, double h, 
								            bool bX, bool bY, bool bZ)
{
    // get id of neighbors with periodic boundary conditions
    // and no-flux conditions
    int xlid,xrid,ylid,yrid,zlid,zrid;
    // -----------------------------------
    // X-Direction Boundaries
    // -----------------------------------
    if (bX) {
        // PBCs (x-dir.)
        if(x == 0) xlid = nx*ny*z + nx*y + nx-1;
        else xlid = nx*ny*z + nx*y + x-1;
        if(x == nx-1) xrid = nx*ny*z + nx*y + 0;
        else xrid = nx*ny*z + nx*y + x+1;
    }
    else {
        // no-flux BC (x-dir.)
		if (x == 0) xlid = nx*ny*z + nx*y + x;
		else xlid = nx*ny*z + nx*y + x-1;
		if (x == nx-1) xrid = nx*ny*z + nx*y + x;
		else xrid = nx*ny*z + nx*y + x+1;
    }
    // -----------------------------------
    // Y-Direction Boundaries
    // -----------------------------------
	if (bY) {
        // PBC Apply
        if(y == 0) ylid = nx*ny*z + nx*(ny-1) + x;
    	else ylid = nx*ny*z + nx*(y-1) + x;
    	if(y == ny-1) yrid = nx*ny*z + nx*0 + x;
    	else yrid = nx*ny*z + nx*(y+1) + x;
    }
    else {
   	// no-flux BC (y-dir.)
        if(y == 0) ylid = nx*ny*z + nx*y + x;
    	else ylid = nx*ny*z + nx*(y-1) + x;
    	if(y == ny-1) yrid = nx*ny*z + nx*y + x;
    	else yrid = nx*ny*z + nx*(y+1) + x;
    }
    // -----------------------------------
    // Z-Direction Boundaries
    // -----------------------------------
	if (bZ) {
		// PBC Apply (z-dir.)
   	if(z == 0) zlid = nx*ny*(nz-1) + nx*y + x;
    	else zlid = nx*ny*(z-1) + nx*y + x;
    	if(z == nz-1) zrid = nx*ny*0 + nx*y + x;
    	else zrid = nx*ny*(z+1) + nx*y + x;
    }
	else {
		// no-flux BC (z-dir.)
		if(z == 0) zlid = nx*ny*z + nx*y + x;
    	else zlid = nx*ny*(z-1) + nx*y + x;
    	if(z == nz-1) zrid = nx*ny*z + nx*y + x;
    	else zrid = nx*ny*(z+1) + nx*y + x;
	}
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


/*************************************************************
  * Compute diffusive interaction parameter in x-direction
  ***********************************************************/

__device__ double chiDiffuse(double locWater, double chiPS, double chiPN)
{
	//int idx = blockIdx.x*blockDim.x + threadIdx.x;
    //double water_diff = (water_CB-0.0)*erfc((idx)/(2.0*sqrt(chiCond*double(current_step)*dt)))+ 0.0;
    double chi = chiPN*locWater + chiPS*(1.0-locWater);
	return chi;
}

/*__device__ double waterDiff(double water_CB, int current_step, double dt,double chiCond)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
    double water_diff = (water_CB-0.0)*erfc((idx)/(2.0*sqrt(chiCond*double(current_step)*dt)))+ 0.0;
	return water_diff;
}*/



/*************************************************************
	* Compute the chemical potential using the 1st derivative
	* of the  binary Flory-Huggins free energy of mixing with
	* respect to c
	*
	* F = c*log(c)/N + (1-c)*log(1-c) + chi*c*(1-c)
	*
	*
	* dF/dc = (log(c) + 1)/N - log(1 - c) - 1.0 
	*         + chi*(1 - 2*c)
	*
	***********************************************************/

__device__ double freeEnergyBiFH(double cc, double chi, double N, double lap_c, double kap, double A)
{
   double c_fh = 0.0;
   if (cc < 0.0) c_fh = 0.0001;
   else if (cc > 1.0) c_fh = 0.999;
   else c_fh = cc;
   double FH = (log(c_fh) + 1.0)/N - log(1.0-c_fh) - 1.0 + chi*(1.0-2.0*c_fh) - kap*lap_c;
   if (cc <= 0.0) FH = -1.5*A*sqrt(-cc) - kap*lap_c;   
   return FH;
}

/*************************************************************
  * Compute second derivative of FH with respect to phi
  ***********************************************************/
  
__device__ double d2dc2_FH(double cc, double N)
{
   double c2_fh = 0.0;
   if (cc < 0.0) c2_fh = 0.0001;
   else if (cc > 1.0) c2_fh = 0.999;
   else c2_fh = cc;
   double FH_2 = 0.5 * (1.0/(N*c2_fh) + 1.0/(1.0-c2_fh));
   return FH_2;	
}

/*************************************************************
  * Compute diffusion coefficient via phillies eq.
  ***********************************************************/

__device__ double philliesDiffusion(double cc, double gamma, double nu, 
								    double D0, double Mweight, double Mvolume)
{
	double cc_d = 1.0;
	double rho = Mweight/Mvolume;
	if (cc >= 1.0) cc_d = 1.0 * rho; // convert phi to g/L	
	else if (cc < 0.0) cc_d = 0.0001 * rho; // convert phi to g/L 
	else cc_d = cc * rho; // convert phi to g/L
	double Dp = D0 * exp(-gamma * pow(cc_d,nu));
	return Dp;
}


// -------------------------------------------------------
// Device Kernels for Testing
// -------------------------------------------------------


/****************************************************************
  * Kernels for unit testing the laplacian devices 
  ***************************************************************/

__global__ void testLapSIPS(double* f, int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        f[gid] = laplacianUpdateBoundaries(f,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
    }
}

__global__ void testLapNonUniformMob(double* f, double *Mob, int nx, int ny, int nz, double h, bool bX, bool bY, bool bZ)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        f[gid] = laplacianNonUniformMob(f,Mob,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
    }
}




// -------------------------------------------------------
// Device Kernels for Simulation
// -------------------------------------------------------


/*********************************************************
  * Compute the laplacian of the concentration array c and w
  * and store it in the device array df and wdf
  *******************************************************/

__global__ void calculateLapBoundaries(double* c,double* df,/*double* w,double* wdf,*/ int nx, int ny, int nz, 
													double h, bool bX, bool bY, bool bZ)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        df[gid] = laplacianUpdateBoundaries(c,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
    }
}



/*********************************************************
  * Computes the chemical potential of a concentration
  * order parameter and stores it in the df_d array.
  *******************************************************/


__global__ void calculateChemPotFH(double* c,double* w,double* df, double kap, double A, double chiPS, double chiPN, double N, int nx, int ny, int nz, int current_step, double dt)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        double cc = c[gid];
        double ww = w[gid];
        double lap_c = df[gid];
        // compute interaction parameter
        double chi = chiDiffuse(ww,chiPS,chiPN);
        // compute chemical potential
        df[gid] = freeEnergyBiFH(cc,chi,N,lap_c,kap,A); 
    }
}


/*********************************************************
  * Computes the mobility of a concentration order
  * parameter and stores it in the Mob_d array.
  *******************************************************/
  
__global__ void calculateMobility(double* c,double* Mob, double M,double mobReSize, int nx, int ny, int nz,
											 double phiCutoff, double N,
        									 double gamma, double nu, double D0, double Mweight, double Mvolume, double Tcast)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        M = 1.0;
        int gid = nx*ny*idz + nx*idy + idx;
        double cc = c[gid];
        double FH2 = d2dc2_FH(cc,N);
        double D_phil = philliesDiffusion(cc,gamma,nu,D0,Mweight,Mvolume);
        double Dtemp = D0*Tcast/273.15;
        M = Dtemp*D_phil/FH2;
        if (M > 1.0) M = 1.0;     // making mobility max = 1
        else if (M < 0.0) M = 0.001; // mobility min = 0.001
        // Using phiCutoff as vitrification
        if (cc > phiCutoff) { 
            M *= 1e-6;
        }
        // resize mobility to be similar to experiments
        M *= mobReSize;
        Mob[gid] = M;		  
    }
}

/************************************************************************************
  * Computes the non-uniform mobility and chemical potential laplacian, multiplies 
  * it by the time step to get the RHS of the CH equation, then uses this RHS value 
  * to perform an Euler update of the concentration in time.
  ***********************************************************************************/

__global__ void lapChemPotAndUpdateBoundaries(double* c,double* df,double* Mob,double* nonUniformLap, double dt, int nx, int ny, int nz, double h,bool bX, bool bY, bool bZ)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        // compute chemical potential laplacain with non-uniform mobility
        // and user defined boundaries (no-flux or PBCs)
        nonUniformLap[gid] = laplacianNonUniformMob(df,Mob,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
        c[gid] += nonUniformLap[gid]*dt;
    } 
}

/*__global__ void diffuseWater(double* w,double* wdf,int nx,int ny,int nz,double h,bool bX,bool bY,bool bZ)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        wdf[gid] = laplacianUpdateBoundaries(w,gid,idx,idy,idz,nx,ny,nz,h,bX,bY,bZ);
    }   
}*/
__global__ void updateWater(double* w,double* wdf,double water_CB,double Dw,double dt,int nx,int ny,int nz)
{
        // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        if (idx == 0) w[gid] = water_CB;
        else w[gid] += Dw*wdf[gid]*dt;   
    }
}
// for calculating water concentration and chi concentration
/*__global__ void calculateWaterChi(double *w, double *chi, int nx, int ny, int nz, double water_CB, int current_step, double dt, double chiCond, double chiPN, double chiPS)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
            int gid = nx*ny*idz + nx*idy + idx;
        	int idx = blockIdx.x*blockDim.x + threadIdx.x;
            double water_diff = (water_CB-0.0)*erfc((idx)/(2.0*sqrt(chiCond*double(current_step)*dt)))+ 0.0;
	        w[gid] = water_diff;
            double chiPS_diff = chiPN*water_diff + chiPS*(1.0-water_diff);
            chi[gid] = chiPS_diff;
    }
}*/
//(w_d,chi_d,nx,ny,nz,water_CB,current_step,dt,chiCond);



/**********************************************************************
  * initialize cuRAND for thermal fluctuations of polymerconcentration
  *********************************************************************/
__global__ void init_cuRAND(unsigned long seed,curandState *state,int nx,int ny,int nz)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        curand_init(seed,gid,0,&state[gid]);
    }
}


/************************************************************
  * Add random fluctuations for non-trivial solution (cuRand)
  ***********************************************************/
__global__ void addNoise(double *c,int nx, int ny, int nz, double dt, int current_step,double chiCond, 
                         double water_CB,double phiCutoff,curandState *state)
{
    // get unique thread id
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if (idx<nx && idy<ny && idz<nz)
    {
        int gid = nx*ny*idz + nx*idy + idx;
        // double water_cutoff = waterDiff(water_CB,current_step,dt,chiCond);
        double noise = curand_uniform_double(&state[gid]);
        double cc = c[gid];
        double noiseScale = 1.0;
        // add random fluctuations with euler update
        if (cc > phiCutoff) noise = 0.5; // no fluctuations for phi < 0
        else if (cc < 0.0) noise = 0.5;  // no fluctuations for phi > phiCutoff
        // ------------------------------------------------------
        // TODO scaling noise
        // ------------------------------------------------------
        // need to minimise noise effect on vitrified morphology
        // testing different methods
        // ------------------------------------------------------
        // stepwise decrease in noise
        // if (water_cutoff > 0.3) noise = 0.5;  
        // scaling noise based off of water concentration 
        // using a simple exponential function
        /*if (water_cutoff > 0.3) {
            double noise_xNorm = (water_cutoff - 0.3)/(water_CB-0.3);
            double noiseScale = 1.0 * exp(-10.0*noise_xNorm);
        }*/
        // scaling noise similar to mobility scaling
        /*if (water_cutoff > 0.30) {
            double xWnorm = (water_cutoff - 0.30)/(water_CB-0.30);
            double noiseScale = 1.0 - (0+((1.0 - 0.0)/(1.0 + exp(-10.0*(xWnorm-(0.0+1.0)/2)))));
        }*/
        c[gid] += 0.1*(noise-0.5)*dt*noiseScale;
    }
}


/*********************************************************
  * Copies the contents of c into cpyBuffer so the c data
  * can be asynchronously transfered from the device to
  * the host.
  *******************************************************/

__global__ void populateCopyBufferSIPS(double* c,double* cpyBuff, int nx, int ny, int nz)
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
