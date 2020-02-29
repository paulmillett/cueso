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
    c2 = input_params("PFSips/c2",0.20);
    c3 = input_params("PFSips/c3",0.20);
    r1 = input_params("PFSips/r1",1.0);
    r2 = input_params("PFSips/r2",0.0);
    NS_depth = input_params("PFSips/NS_depth",10);
    M = input_params("PFSips/M",1.0);
    mobReSize = input_params("PFSips/mobReSize",0.35);
    kap = input_params("PFSips/kap",1.0);
    water_CB = input_params("PFSips/water_CB",1.0);
    mobReSize = input_params("PFSips/mobReSize",0.35);
    chiPS = input_params("PFSips/chiPS",0.034);
    chiPN = input_params("PFSips/chiPN",1.5);
    chiCond = input_params("PFSips/chiCond",0);
    chiFreeze = input_params("PFSips/chiFreeze",1.75);
    phiCutoff = input_params("PFSips/phiCutoff",0.75);
    N = input_params("PFSips/N",100.0);
    A = input_params("PFSips/A",1.0);
    Tinit = input_params("PFSips/Tinit",298.0);
    Tcast = input_params("PFSips/Tcast",298.0);
    noiseStr = input_params("PFSips/noiseStr",0.1);
    D0 = input_params("PFSips/D0",1.0);
    Dw = input_params("PFSips/Dw",1.0);
    nu = input_params("PFSips/nu",1.0);
    gamma = input_params("PFSips/gamma",1.0);
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
    // just for copyin water and chi concentrations
    cudaFree(w_d);
    //cudaFree(wdf_d);
    // cudaFree(chi_d);
    cudaFree(df_d);
    cudaFree(cpyBuff_d);
    cudaFree(Mob_d);
    cudaFree(nonUniformLap_d);
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
	 // random initialization
    int xHolder = 0;
    int zone1 = r1*(nx-NS_depth); 
    int zone2 = r2*(nx-NS_depth);
    int zone3 = nx - zone1 - zone2 - NS_depth; 
    for(size_t i=0;i<nxyz;i++) {
        double r = (double)rand()/RAND_MAX;
        // create NonSolvent layer
        while (xHolder < NS_depth) 
        { 
            water.push_back(water_CB);
            c.push_back(0.0);
            xHolder++;
        }
        xHolder = 0;
        // initialize first polymer layer
        while (xHolder < zone1) 
        {
            r = (double)rand()/RAND_MAX; 
            c.push_back(co + 0.1*(r-0.5)); 
            water.push_back(0.0);
            xHolder++;
        }
        xHolder = 0;
        // initialize second polymer layer
        while (xHolder < zone2) 
        {
            r = (double)rand()/RAND_MAX; 
            c.push_back(c2 + 0.1*(r-0.5));
            water.push_back(0.0);
            xHolder++;
        }
        xHolder = 0;
        // initialize third polymer layer
        while (xHolder < zone3) 
        {
            r = (double)rand()/RAND_MAX; 
            c.push_back(c3 + 0.1*(r-0.5));
            water.push_back(0.0);
            xHolder++;
        }
        xHolder = 0;
    }
    
    // initializing water and chi arrays for printout
    /*for(size_t i=0; i<nxyz; i++)
    {
        //chi.push_back(0.0);
        //water.push_back(0.0);
    }*/
    
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
    // cudaMalloc((void**) &wdf_d,size);
    // cudaCheckErrors("cudaMalloc fail");
    cudaMalloc((void**) &cpyBuff_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate mobility
    cudaMalloc((void**) &Mob_d,size);
    cudaCheckErrors("cudaMalloc fail");
    // allocate nonuniform laplacian for mobility
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
        // TODO
        
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
        
        // calculate laplacian for diffusing water
        calculateLapBoundaries<<<blocks,blockSize>>>(w_d,df_d,nx,ny,nz,dx,bx,by,bz);
        cudaCheckAsyncErrors('calculateLap water kernel fail');    
        cudaDeviceSynchronize();
        
        // euler update water diffusing
        updateWater<<<blocks,blockSize>>>(w_d,df_d,water_CB,Dw,dt,nx,ny,nz);
        cudaCheckAsyncErrors("updateWater kernel fail");
        cudaDeviceSynchronize();
        // TODO 
        // need to update w_d with an euler update...
        
        
        // add thermal fluctuations of polymer concentration
        addNoise<<<blocks,blockSize>>>(c_d, nx, ny, nz, dt, current_step, chiCond, water_CB, phiCutoff, devState);
        cudaCheckAsyncErrors("addNoise kernel fail");
        cudaDeviceSynchronize();
        
        // calculate w and chi
        //calculateWaterChi<<<blocks,blockSize>>>(w_d,chi_d,nx,ny,nz,water_CB,current_step,dt,chiCond,chiPN,chiPS);
        //cudaCheckAsyncErrors("calculateWaterChi kernel fail");
        //cudaDeviceSynchronize();
        
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
    // interaction parameter concentration
    //populateCopyBufferSIPS<<<blocks,blockSize>>>(chi_d,cpyBuff_d,nx,ny,nz);
    //cudaMemcpyAsync(&chi[0],chi_d,size,cudaMemcpyDeviceToHost);
    //cudaCheckErrors("cudaMemcpyAsync D2H fail");
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
    ofstream outfile3;
    stringstream filenamecombine;
    stringstream filenamecombine2;
    stringstream filenamecombine3;
    
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
    outfile2 << "SCALARS c float" << endl;
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
                //if (point < 1e-10) point = 0.0; // making really small numbers == 0 
                outfile2 << point << endl;
            }

    // -----------------------------------
    //	Close the file:
    // -----------------------------------

    outfile2.close();
    
    
    // write output for chi
    // -----------------------------------
    // Define the file location and name:
    // -----------------------------------

    /*filenamecombine3 << "vtkoutput/chi_" << step << ".vtk";
    string filename3 = filenamecombine3.str();
    outfile3.open(filename3.c_str(), std::ios::out);

    // -----------------------------------
    //	Write the 'vtk' file header:
    // -----------------------------------

    outfile3 << "# vtk DataFile Version 3.1" << endl;
    outfile3 << "VTK file containing grid data" << endl;
    outfile3 << "ASCII" << endl;
    outfile3 << " " << endl;
    outfile3 << "DATASET STRUCTURED_POINTS" << endl;
    outfile3 << "DIMENSIONS" << d << nx << d << ny << d << nz << endl;
    outfile3 << "ORIGIN " << d << 0 << d << 0 << d << 0 << endl;
    outfile3 << "SPACING" << d << 1.0 << d << 1.0 << d << 1.0 << endl;
    outfile3 << " " << endl;
    outfile3 << "POINT_DATA " << nxyz << endl;
    outfile3 << "SCALARS c float" << endl;
    outfile3 << "LOOKUP_TABLE default" << endl;

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
                double point = chi[id];
                //if (point < 1e-10) point = 0.0; // making really small numbers == 0 
                outfile3 << point << endl;
            }

    // -----------------------------------
    //	Close the file:
    // -----------------------------------

    outfile3.close();*/
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
