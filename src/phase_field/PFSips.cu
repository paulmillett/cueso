# include <iostream>    // endl
# include <fstream>     // for ofstream
# include <string>      // for string
# include <sstream>     // for stringstream
# include <math.h>
# include "PFSips.h"
# include "PFSipsKernels.h"
# include "../utils/cudaErrorMacros.h" // for cudaCheckErrors & cudaCheckAsyncErrors


using std::string;
using std::stringstream;
using std::cout;
using std::endl;
using std::ofstream;

// -------------------------------------------------------------------------
// Constructor:
// -------------------------------------------------------------------------

PFSips::PFSips(const GetPot& input_params)
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
    bx = input_params("PFSips/bx",0);
    by = input_params("PFSips/by",1);
    bz = input_params("PFSips/bz",1);
    numSteps = input_params("Time/nstep",1);
    co = input_params("PFSips/co",0.20);
    M = input_params("PFSips/M",1.0);
    mobReSize = input_params("PFSips/mobReSize",0.35);
    kap = input_params("PFSips/kap",1.0);
    water_CB = input_params("PFSips/water_CB",1.0);
    NS_in_dope = input_params("PFSips/NS_in_dope",0.0);
    mobReSize = input_params("PFSips/mobReSize",0.35);
    chiPS = input_params("PFSips/chiPS",0.034);
    chiPN = input_params("PFSips/chiPN",1.5);
    phiCutoff = input_params("PFSips/phiCutoff",0.75);
    N = input_params("PFSips/N",100.0);
    A = input_params("PFSips/A",1.0);
    Tinit = input_params("PFSips/Tinit",298.0);
    Tcast = input_params("PFSips/Tcast",298.0);
    noiseStr = input_params("PFSips/noiseStr",0.1);
    D0 = input_params("PFSips/D0",1.0);
    Dw = input_params("PFSips/Dw",1.0);
    nu = input_params("PFSips/nu",1.0);
    nuDw = input_params("PFSips/nuDw",1.0);
    gamma = input_params("PFSips/gamma",1.0);
    gammaDw = input_params("PFSips/gammaDw",1.0);
    Mweight = input_params("PFSips/Mweight",100.0);
    Mvolume = input_params("PFSips/Mvolume",0.1);
    numOutputs = input_params("Output/numOutputs",1);
    outInterval = numSteps/numOutputs;
    // ---------------------------------------
    // Set up cuda kernel launch variables:
    // ---------------------------------------

    blockSize.x = input_params("GPU/blockSize.x",0);
    blockSize.y = input_params("GPU/blockSize.y",0);
    blockSize.z = input_params("GPU/blockSize.z",0);

    // set default kernel launch parameters
    if(blockSize.x == 0) blockSize.x = 32;
    if(blockSize.y == 0) blockSize.y = 32;
    if(blockSize.z == 0) blockSize.z = 1;

    // calculate the number of blocks to be used (3-D block grid)
    int totalBlockSize = blockSize.x*blockSize.y*blockSize.z;
    blocks.x = (nx + blockSize.x - 1)/blockSize.x;
    blocks.y = (ny + blockSize.y - 1)/blockSize.y;
    blocks.z = (nz + blockSize.z - 1)/blockSize.z;

    // perform some assumption checking
    int numBlocks = blocks.x*blocks.y*blocks.z;
    int totalNumThreads = numBlocks*totalBlockSize;
    if(totalNumThreads < nxyz)
        throw "GPU Kernel Launch setup lacks sufficient threads!\n";
    if(totalBlockSize > 1024)
        throw "Total number of threads per block exceeds 1024";

}



// -------------------------------------------------------------------------
// Destructor:
// -------------------------------------------------------------------------

PFSips::~PFSips()
{

    // ----------------------------------------
    // free up device memory:
    // ----------------------------------------

    cudaFree(c_d);
    cudaFree(df_d);
    cudaFree(Mob_d);
    cudaFree(w_d);
    cudaFree(muNS_d);
    cudaFree(nonUniformLap_d);
    cudaFree(cpyBuff_d);
    cudaFree(devState);
}



// -------------------------------------------------------------------------
// Initialize system:
// -------------------------------------------------------------------------

void PFSips::initSystem()
{
		
    // ----------------------------------------
    // Initialize concentration fields:
    // ----------------------------------------
    srand(time(NULL));      // setting the seed  
    double r = 0.0;
    for(size_t i=0;i<nxyz;i++) {
        r = (double)rand()/RAND_MAX;
        // initialize polymer phase
        c.push_back(co + 0.1*(r-0.5));
        // initialize nonsolvent phase
        water.push_back(NS_in_dope);
    }

    // ----------------------------------------
    // Allocate memory on device and copy data
    // and copy data from host to device
    // ----------------------------------------

    // allocate memory on device
    size = nxyz*sizeof(double);
    // allocate polymer species
    cudaMalloc((void**) &c_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate space for laplacian
    cudaMalloc((void**) &df_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate water concentration
    cudaMalloc((void**) &w_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate space for laplacian
    cudaMalloc((void**) &muNS_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // copy buffer
    cudaMalloc((void**) &cpyBuff_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate mobility
    cudaMalloc((void**) &Mob_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate nonuniform laplacian for mobility 
    // and water diffusion coefficient
    cudaMalloc((void**) &nonUniformLap_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate memory for cuRAND state
    cudaMalloc((void**) &devState,nxyz*sizeof(curandState));
    cudaCheckErrors("cudaMalloc fail");
    // copy concentration and water array to device
    cudaMemcpy(c_d,&c[0],size,cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D fail");
    cudaMemcpy(w_d,&water[0],size,cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D fail");
    
    // ----------------------------------------
    // Initialize thermal fluctuations of
    // polymer concentration
    // ----------------------------------------
    
    init_cuRAND<<<blocks,blockSize>>>(time(NULL),devState,nx,ny,nz);
    
}



// -------------------------------------------------------------------------
// Take one step forward in time:
// -------------------------------------------------------------------------

void PFSips::computeInterval(int interval)
{

    // ----------------------------------------
    //	Set the time step:
    // ----------------------------------------

    current_step = interval*outInterval;

    // ----------------------------------------
    //	Evolve system by solving CH equation:
    // ----------------------------------------

    for(size_t i=0;i<outInterval;i++)
    {
        // calculate the laplacian of c_d and store in df_d
        calculateLapBoundaries<<<blocks,blockSize>>>(c_d,df_d,nx,ny,nz,dx,bx,by,bz); 
        cudaCheckAsyncErrors("calculateLap polymer kernel fail");
        cudaDeviceSynchronize();
        
        // calculate the chemical potential and store in df_d
        calculateChemPotFH<<<blocks,blockSize>>>(c_d,w_d,df_d,kap,A,chiPS,chiPN,N,nx,ny,nz,current_step,dt);
        cudaCheckAsyncErrors("calculateChemPotFH kernel fail");
        cudaDeviceSynchronize();
        
        // calculate mobility and store it in Mob_d
        calculateMobility<<<blocks,blockSize>>>(c_d,Mob_d,M,mobReSize,nx,ny,nz,phiCutoff,N,gamma,nu,D0,Mweight,Mvolume,Tcast);
        cudaCheckAsyncErrors("calculateMobility kernel fail");
        cudaDeviceSynchronize();

        // calculate the laplacian of the chemical potential, then update c_d
        // using an Euler update
        lapChemPotAndUpdateBoundaries<<<blocks,blockSize>>>(c_d,df_d,Mob_d,nonUniformLap_d, dt,nx,ny,nz,dx,bx,by,bz);
        cudaCheckAsyncErrors("lapChemPotAndUpdateBoundaries kernel fail");
        cudaDeviceSynchronize();
        
        // calculate mu for Nonsolvent diffusion
        calculate_muNS<<<blocks,blockSize>>>(w_d,c_d,muNS_d,Mob_d,Dw,water_CB,gammaDw,nuDw,Mweight,Mvolume,nx,ny,nz);
        cudaCheckAsyncErrors('calculate muNS kernel fail');
        cudaDeviceSynchronize();
        
        // calculate laplacian for diffusing water
        calculateLapBoundaries_muNS<<<blocks,blockSize>>>(df_d,muNS_d,nx,ny,nz,dx,bx,by,bz);
        cudaCheckAsyncErrors('calculateLap water kernel fail');    
        cudaDeviceSynchronize();
        
        // calculate nonuniform laplacian for diffusion
        calculateNonUniformLapBoundaries_muNS<<<blocks,blockSize>>>(muNS_d,Mob_d,nonUniformLap_d,nx,ny,nz,dx,bx,by,bz);
        cudaCheckAsyncErrors('calculateNonUniformLap muNS kernel fail');
        cudaDeviceSynchronize();
        
        // euler update water diffusing
        update_water<<<blocks,blockSize>>>(w_d,df_d,Mob_d,nonUniformLap_d,dt,nx,ny,nz,dx,bx,by,bz);
        cudaCheckAsyncErrors("updateWater kernel fail");
        cudaDeviceSynchronize();
        
        // add thermal fluctuations of polymer concentration
        addNoise<<<blocks,blockSize>>>(c_d, nx, ny, nz, dt, current_step, water_CB, phiCutoff, devState);
        cudaCheckAsyncErrors("addNoise kernel fail");
        cudaDeviceSynchronize(); 
    }

    // ----------------------------------------
    //	Copy data back to host for writing:
    // ----------------------------------------
    
    // polymer concentration
    populateCopyBufferSIPS<<<blocks,blockSize>>>(c_d,cpyBuff_d,nx,ny,nz);
    cudaMemcpyAsync(&c[0],c_d,size,cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpyAsync D2H fail");
    cudaDeviceSynchronize();
    // nonsolvent concentration
    populateCopyBufferSIPS<<<blocks,blockSize>>>(w_d,cpyBuff_d,nx,ny,nz);
    cudaMemcpyAsync(&water[0],w_d,size,cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpyAsync D2H fail");
    cudaDeviceSynchronize();
}



// -------------------------------------------------------------------------
// Write output:
// -------------------------------------------------------------------------

void PFSips::writeOutput(int step)
{

    // -----------------------------------
    // Define the file location and name:
    // -----------------------------------

    ofstream outfile;
    ofstream outfile2;
    stringstream filenamecombine;
    stringstream filenamecombine2;
    
    filenamecombine << "vtkoutput/c_" << step << ".vtk";
    string filename = filenamecombine.str();
    outfile.open(filename.c_str(), std::ios::out);

    // -----------------------------------
    //	Write the 'vtk' file header:
    // -----------------------------------

    string d = "   ";
    outfile << "# vtk DataFile Version 3.1" << endl;
    outfile << "VTK file containing grid data" << endl;
    outfile << "ASCII" << endl;
    outfile << " " << endl;
    outfile << "DATASET STRUCTURED_POINTS" << endl;
    outfile << "DIMENSIONS" << d << nx << d << ny << d << nz << endl;
    outfile << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
    outfile << "SPACING" << d << 1.0 << d << 1.0 << d << 1.0 << endl;
    outfile << " " << endl;
    outfile << "POINT_DATA " << nxyz << endl;
    outfile << "SCALARS c float" << endl;
    outfile << "LOOKUP_TABLE default" << endl;

    // -----------------------------------
    //	Write the data:
    // NOTE: x-data increases fastest,
    //       then y-data, then z-data
    // -----------------------------------

    for(size_t k=0;k<nz;k++)
        for(size_t j=0;j<ny;j++)
            for(size_t i=0;i<nx;i++)
            {
                int id = nx*ny*k + nx*j + i;
                double point = c[id];
                //if (point < 1e-10) point = 0.0; // making really small numbers == 0 
                outfile << point << endl;
            }

    // -----------------------------------
    //	Close the file:
    // -----------------------------------

    outfile.close();
    // vtkoutput for water
    // -----------------------------------
    // Define the file location and name:
    // -----------------------------------


    filenamecombine2 << "vtkoutput/w_" << step << ".vtk";
    string filename2 = filenamecombine2.str();
    outfile2.open(filename2.c_str(), std::ios::out);

    // -----------------------------------
    //	Write the 'vtk' file header:
    // -----------------------------------

    outfile2 << "# vtk DataFile Version 3.1" << endl;
    outfile2 << "VTK file containing grid data" << endl;
    outfile2 << "ASCII" << endl;
    outfile2 << " " << endl;
    outfile2 << "DATASET STRUCTURED_POINTS" << endl;
    outfile2 << "DIMENSIONS" << d << nx << d << ny << d << nz << endl;
    outfile2 << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
    outfile2 << "SPACING" << d << 1.0 << d << 1.0 << d << 1.0 << endl;
    outfile2 << " " << endl;
    outfile2 << "POINT_DATA " << nxyz << endl;
    outfile2 << "SCALARS w float" << endl;
    outfile2 << "LOOKUP_TABLE default" << endl;

    // -----------------------------------
    //	Write the data:
    // NOTE: x-data increases fastest,
    //       then y-data, then z-data
    // -----------------------------------

    for(size_t k=0;k<nz;k++)
        for(size_t j=0;j<ny;j++)
            for(size_t i=0;i<nx;i++)
            {
                int id = nx*ny*k + nx*j + i;
                double point = water[id];
                // for paraview
                if (point < 1e-30) point = 0.0; // making really small numbers == 0 
                outfile2 << point << endl;
            }

    // -----------------------------------
    //	Close the file:
    // -----------------------------------

    outfile2.close();
    
}



// -------------------------------------------------------------------------
// Run unit tests for this App:
// -------------------------------------------------------------------------

void PFSips::runUnitTests()
{
    bool pass;
    pass = lapKernUnitTest();
    if(pass)
        cout << "\t- lapKernUnitTest -------------- PASSED\n";
    else
        cout << "\t- lapKernUnitTest -------------- FAILED\n";
}



// -------------------------------------------------------------------------
// Unit tests for this App:
// -------------------------------------------------------------------------

bool PFSips::lapKernUnitTest()
{
    // 3X3X3 scalar field with ones except the central node
    double sf[27] = {1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1};
    double solution[27] = {0,0,0,0,-1,0,0,0,0,0,-1,0,-1,6,-1,0,-1,0,0,0,0,0,-1,0,0,0,0};
    // allocate space on device
    double* sf_d;
    cudaMalloc((void**) &sf_d,27*sizeof(double));
    cudaCheckErrors("cudaMalloc fail");
    // copy sf to device
    cudaMemcpy(sf_d,sf,27*sizeof(double),cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D fail");
    // launch kernel
    dim3 grid(1,1,3);
    dim3 TpB(32,32,1);
    testLapSIPS<<<grid,TpB>>>(sf_d,3,3,3,1.0,bx,by,bz);
    // copy data back to host
    cudaMemcpy(sf,sf_d,27*sizeof(double),cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H fail");
    // print out results
    for(size_t i=0;i<27;i++)
        /* cout << "i=" << i << " sf=" << sf[i] << " sol=" << solution[i] << endl; */
        if( sf[i] != solution[i]) 
        {
            cout << "i=" << i << " sf=" << sf[i] << " sol=" << solution[i] << endl;
            return false;
        }
    return true;
}
