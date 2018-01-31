
# include "PFApp.h"
# include "PFAppKernels.h"
# include "../utils/cudaErrorMacros.h" // for cudaCheckErrors & cudaCheckAsyncErrors

// -------------------------------------------------------------------------
// Constructor:
// -------------------------------------------------------------------------

PFApp::PFApp(const GetPot& input_params)
    : rng(1234)
{

    // ---------------------------------------
    // Assign variables from 'input_params':
    // ---------------------------------------

    nx = input_params("Domain/nx",1);
    ny = input_params("Domain/ny",1);
    nz = input_params("Domain/nz",1);
    nxyz = nx*ny*nz;
    dx = input_params("Domain/dx",1.0);
    dy = input_params("Domain/dy",1.0);
    dz = input_params("Domain/dz",1.0);
    dt = input_params("Time/dt",1.0);
    co = input_params("PFApp/co",0.5);
    M = input_params("PFApp/M",1.0);
    w = input_params("PFApp/w",1.0);
    kap = input_params("PFApp/kap",1.0);
    numOutputs = input_params("Output/numOutputs",1);
    numSteps = input_params("Time/nstep",1);
    outInterval = numSteps/numOutputs;

    // ---------------------------------------
    // Set up cuda kernel launch variables:
    // ---------------------------------------

    blocks.x = input_params("GPU/blocks.x",0);
    blocks.y = input_params("GPU/blocks.y",0);
    blocks.z = input_params("GPU/blocks.z",0);
    blockSize.x = input_params("GPU/blockSize.x",0);
    blockSize.y = input_params("GPU/blockSize.y",0);
    blockSize.z = input_params("GPU/blockSize.z",0);

    // set default kernel launch parameters
    if(blocks.x == 0) blocks.x = 1;
    if(blocks.y == 0) blocks.y = 1;
    if(blocks.z == 0) blocks.z = 1;
    if(blockSize.x == 0) blockSize.x = 32;
    if(blockSize.y == 0) blockSize.y = 32;
    if(blockSize.z == 0) blockSize.z = 1;
    
    // perform some assumption checking
    int numBlocks = blocks.x*blocks.y*blocks.z;
    int totalBlockSize = blockSize.x*blockSize.y*blockSize.z;
    int totalNumThreads = numBlocks*totalBlockSize;
    if(totalNumThreads < nxyz)
        throw "GPU Kernel Launch setup lacks sufficient threads!\n";
    if(totalBlockSize > 1024)
        throw "Total number of threads per block exceeds 1024";

}



// -------------------------------------------------------------------------
// Destructor:
// -------------------------------------------------------------------------

PFApp::~PFApp()
{

    // ----------------------------------------
    // free up device memory:
    // ----------------------------------------

    cudaFree(c_d);
    cudaFree(df_d);
}



// -------------------------------------------------------------------------
// Initialize system:
// -------------------------------------------------------------------------

void PFApp::initSystem()
{

    // ----------------------------------------
    // Initialize the concentration field:
    // ----------------------------------------

    for(size_t i=0;i<nxyz;i++)
        c.push_back(co + 0.1*(rng.uniform()-0.5));

    // ----------------------------------------
    // Allocate memory on device and copy data
    // and copy data from host to device
    // ----------------------------------------

    // allocate memory on device
    int size = nxyz*sizeof(double);
    cudaMalloc((void**) &c_d,size);
    cudaCheckErrors("cudaMalloc fail");
    cudaMalloc((void**) &df_d,size);
    cudaCheckErrors("cudaMalloc fail");

    // copy concentration array to device
    cudaMemcpy(c_d,&c[0],size,cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D fail");

}



// -------------------------------------------------------------------------
// Take one step forward in time:
// -------------------------------------------------------------------------

void PFApp::computeInterval(int interval)
{

    // ----------------------------------------
    //	Set the time step:
    // ----------------------------------------

    current_step = interval*outInterval;

    // ----------------------------------------
    //	Update CH system:
    // ----------------------------------------

    evolveCH<<<blocks,blockSize>>>(c_d,df_d);
    cudaCheckAsyncErrors("evolveCH kernel fail");

}



// -------------------------------------------------------------------------
// Write output:
// -------------------------------------------------------------------------

void PFApp::writeOutput(int step)
{
}
