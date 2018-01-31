#include <iostream>
#include <random>
#include <fstream>
#include <string>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>

#define nr 512 
#define nc 512 
#define Blk_H 8 
#define Blk_W 8 
#define stclX 1
#define stclY 1
using namespace std;

// cuda code
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

typedef double num_col[nc];

// function declartions
void write_ouput(num_col *phi,int nx,int ny,int step,double dx,double dy);

__global__ void lap_per(num_col * mat_in, num_col * mat_out);

__global__ void calc_df(num_col * mat_in, num_col * mat_out);

__global__ void update_phi(num_col * mat_in, num_col * mat_out);

int main(){
    // declare simulation variables
    int step;
    int steps    = 10000;
    int skip     = 1000;
    int ny       = nr;
    int nx       = nc;
    double dx    = 1.0;
    double dy    = 1.0;
    double phi0  = 0.0;
    num_col * phi;
    num_col * df;
    num_col * d_phi;
    num_col * d_df;

    int size = nx*ny*sizeof(double);

    // allocate resources for phi and df on cpu
    phi = (num_col*) malloc(size);
    df  = (num_col*) malloc(size);

    // allocate memory on gpu
    cudaMalloc((void**) &d_phi,size);
    cudaCheckErrors("cudaMalloc fail");
    cudaMalloc((void**) &d_df,size);
    cudaCheckErrors("cudaMalloc fail");

    // clean up output files from previous runs
//    system("rm fluid*");

    // random number stuff
    default_random_engine generator;
    uniform_real_distribution<double> ran(-1.0,1.0);
    
    // initialize system
    for (int i=0; i<nx;i++){
        for (int j=0; j<ny;j++){
            phi[i][j] = phi0 + 0.1*ran(generator);
        }
    }

    // write initial condition output file
    write_ouput(phi,nx,ny,1,dx,dy);

    // evolve the system using finite difference to
    // solve the CH equation

    // copy data to the gpu
    cudaMemcpy(d_phi,phi,size,cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D fail");
    cudaMemcpy(d_df,df,size,cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D fail");

    // get grid and block size
    dim3 threadsPerBlock(Blk_H,Blk_W);
    dim3 numBlocks( (nr + Blk_H - 1)/threadsPerBlock.x, (nc + Blk_W - 1)/threadsPerBlock.x );

    for (step = 1; step <= steps; step++){

        // pass the number crunching off to the gpu

        // calculate the laplacian of phi and store in df
        lap_per<<<numBlocks,threadsPerBlock>>>(d_phi,d_df);
        cudaDeviceSynchronize();
        cudaCheckErrors("cuda kernel fail");

        // caclulate df and store in df
        calc_df<<<numBlocks,threadsPerBlock>>>(d_phi,d_df);
        cudaDeviceSynchronize();
        cudaCheckErrors("cuda kernel fail");
        
        // calculate the laplacian of df and update phi
        update_phi<<<numBlocks,threadsPerBlock>>>(d_df,d_phi);
        cudaDeviceSynchronize();
        cudaCheckErrors("cuda kernel fail");
        
        // write output
        if (step%skip == 0){
            // get phi from gpu
            cudaMemcpyAsync(phi,d_phi,size,cudaMemcpyDeviceToHost);
            cudaCheckErrors("cudaMemcpy D2H fail");
            write_ouput(phi,nx,ny,step,dx,dy);
        }
    }

    // clean up on device
    cudaFree(d_phi);
    cudaFree(d_df);
    return 0;
}

// function definitions

__global__ void lap_per(num_col * mat_in, num_col * mat_out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int is,in,je,jw;
    int ii,jj,iis,iin,jje,jjw;

    __shared__ double d2x,d2y;

    if (i < nc && j < nr ){

        // get neighbors for stencil for the bulk case
        is = i - 1;
        in = i + 1;
        je = j + 1;
        jw = j - 1;

        // apply periodic boundary conditions
        // south boundary
        if (i == 0) is = nr - 1;
        // north boundary
        if (i == (nr-1)) in = 0;
        // west boundary
        if (j == 0) jw = nc - 1;
        // east boundary
        if (j == (nc-1)) je = 0;

        // shared memory neighbor indices
        ii = threadIdx.x + stclX;
        jj = threadIdx.y + stclY;
        iin = ii + 1;
        iis = ii - 1;
        jje = jj + 1;
        jjw = jj - 1;

        // sync all the threads in the block up to this point
        __syncthreads();

        // memory optimization using shared memory
        __shared__ double smem[(Blk_H + 2*stclX)][(Blk_W + 2*stclY)];
        smem[ii][jj] = mat_in[i][j];
        // load north side of stencil
        if (threadIdx.x > (Blk_H - 2*stclX)){
            smem[ii + stclX][jj] =  mat_in[in][j];
        }
        // load south side of stencil
        if (threadIdx.x == 0){
            smem[ii - stclX][jj] =  mat_in[is][j];
        }
        // load east side of stencil
        if (threadIdx.y > (Blk_W - 2*stclY)){
            smem[ii][jj + stclY] =  mat_in[i][je];
        }
        // load west side of stencil
        if (threadIdx.y == 0){
            smem[ii][jj - stclY] =  mat_in[i][jw];
        }

        // sync threads to ensure all data is loaded into shared memory
        __syncthreads();

        // calculate 2D laplacian approximation (assume dx and dy = 1.0)
        d2y = smem[iis][jj] + smem[iin][jj] - 2.0*smem[ii][jj];
        d2x = smem[ii][jjw] + smem[ii][jje] - 2.0*smem[ii][jj];
        mat_out[i][j] = d2x + d2y;
    }
}

__global__ void calc_df(num_col * mat_in, num_col * mat_out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int ii = threadIdx.x;
    int jj = threadIdx.y;
    if (i < nc && j < nr ){

        // memory optimization using shared memory
        __shared__ double smem[(Blk_H)][(Blk_W)];
        smem[ii][jj] = mat_in[i][j];

        // calulate df[i][j] noting that it currently stores the laplacian of phi
        // also assume kappa=1.0
        mat_out[i][j] = -1.0*mat_out[i][j] + smem[ii][jj]*smem[ii][jj]*smem[ii][jj] - smem[ii][jj];
    }
}

__global__ void update_phi(num_col * mat_in, num_col * mat_out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int is,in,je,jw;
    int ii,jj,iis,iin,jje,jjw;
    __shared__ double d2x,d2y,dt;
    dt = 0.01;
    if (i < nc && j < nr ){
        // get neighbors for stencil for the bulk case
        is = i - 1;
        in = i + 1;
        je = j + 1;
        jw = j - 1;

        // apply periodic boundary conditions
        // left boundary
        if (i == 0) is = nr - 1;
        // right boundary
        if (i == (nr-1)) in = 0;
        // bottom boundary
        if (j == 0) jw = nc - 1;
        // top boundary
        if (j == (nc-1)) je = 0;

        // shared memory neighbor indices
        ii = threadIdx.x + stclX;
        jj = threadIdx.y + stclY;
        iin = ii + 1;
        iis = ii - 1;
        jje = jj + 1;
        jjw = jj - 1;

        // sync all the threads in the block up to this point
        __syncthreads();

        // memory optimization using shared memory
        __shared__ double smem[(Blk_H + 2*stclX)][(Blk_W + 2*stclY)];
        smem[ii][jj] = mat_in[i][j];
        // load north side of stencil
        if (threadIdx.x > (Blk_H - 2*stclX)){
            smem[ii + stclX][jj] =  mat_in[in][j];
        }
        // load south side of stencil
        if (threadIdx.x == 0){
            smem[ii - stclX][jj] =  mat_in[is][j];
        }
        // load east side of stencil
        if (threadIdx.y > (Blk_W - 2*stclY)){
            smem[ii][jj + stclY] =  mat_in[i][je];
        }
        // load west side of stencil
        if (threadIdx.y == 0){
            smem[ii][jj - stclY] =  mat_in[i][jw];
        }

        // sync threads to ensure all data is loaded into shared memory
        __syncthreads();

        // calculate 2D laplacian approximation (assume dx and dy = 1.0)
        d2y = smem[iis][jj] + smem[iin][jj] - 2.0*smem[ii][jj];
        d2x = smem[ii][jjw] + smem[ii][jje] - 2.0*smem[ii][jj];
        mat_out[i][j] = mat_out[i][j] + dt*(d2x + d2y);
    }
}

void write_ouput(num_col *phi,int nx,int ny,int step,double dx,double dy)
{
    string filename = to_string(step);
    filename = "fluid_" + filename + ".vtk";
    string d = "     ";

    ofstream outf(filename);
 
    // If we couldn't open the output file stream for writing
    if (!outf)
    {
        // Print an error and exit
        cerr << "could not open output file for reading!" << endl;
        exit(1);
    }

    // write file header
    outf << "# vtk DataFile Version 3.1" << endl;
    outf << "VTK file containing grid data" << endl;
    outf << "ASCII" << endl;
    outf << endl;
    outf << "DATASET STRUCTURED_POINTS" << endl;
    outf << "DIMENSIONS" << d<< nx << d << ny << "     1" << endl;
    outf << "ORIGIN 1 1 1" << endl;
    outf << "SPACING     1.0     1.0     1.0" << endl;
    outf << endl;
    outf << "POINT_DATA     " <<  nx*ny << endl;
    outf << "SCALARS Phi float" << endl;
    outf << "LOOKUP_TABLE default" << endl;

    // write out the values of phi
    for (int i=0; i<nx;i++){
        for (int j=0; j<ny;j++){
            outf << setw(16) << phi[i][j] << endl;
        }
    }
    
}
