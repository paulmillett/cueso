
# ifndef PFSIPS_H
# define PFSIPS_H

# include <vector>
# include "../base/CuesoBase.h"
# include "../utils/rand.h"
# include "../utils/GetPot"
# include <curand.h>
# include <curand_kernel.h>


using std::vector;

class PFSips : public CuesoBase {

    private:

        int current_step;
        int nx,ny,nz;
        int nxyz;
        int numSteps;
        int numOutputs;
        int outInterval;
        double co;        // polymer concentration fields
        double M;
        double mobReSize;
        double w;
        double kap;
        double dt;
        double dx,dy,dz;
        double water_CB;
        double NS_in_dope;
        double phiCutoff;
        double chiPS;
        double chiPN;
        double N;
        double A;
        double Tbath;
        double Tinit;
        double Tcast;
        double noiseStr;
        double nu;
        double nuDw;
        double gamma;
        double gammaDw;
        double D0;
        double Dw;
        double Mweight;
        double Mvolume;
        bool bx,by,bz;
        Rand rng;
        vector<double> c;
        vector<double> water;
    
        // cuda members
        int size;
        double * c_d;        // concentration array
        double * df_d;       // chemical potential array
        double * w_d;        // non-solvent array
        double * muNS_d;     // laplacian array for fickian diffusion
        double * cpyBuff_d; 			// Copy buffer for ansynchronous data transfer
        double * Mob_d;     			// mobility
        double * nonUniformLap_d;	    // laplacian of mobility and df
        curandState * devState;         // state for cuRAND
        unsigned long seed;             // seed for cuRAND
        dim3 blocks;
        dim3 blockSize;
        

    public:

        PFSips(const GetPot&);
        ~PFSips();
        void initSystem();
        void computeInterval(int interval);
        void writeOutput(int step);
        void runUnitTests();

    private:
        // unit tests
        bool lapKernUnitTest();

};

# endif  // PFSips_H
