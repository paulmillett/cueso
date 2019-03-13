
# ifndef PFSIPS_H
# define PFSIPS_H

# include <vector>
# include "../base/CuesoBase.h"
# include "../utils/rand.h"
# include "../utils/GetPot"


using std::vector;

class PFSips : public CuesoBase {

    private:

        int current_step;
        int nx,ny,nz;
        int nxyz;
        int numSteps;
        int numOutputs;
        int outInterval;
        double co;
        double M;
        double w;
        double kap;
        double dt;
        double dx,dy,dz;
        double chiCond;
        double phiCutoff;
        double chiPS;
        double chiPN;
        double N;
        double alpha;
        double beta;
        double eta;
        double A;
        double Tbath;
        double Tinit;
        double noiseStr;
        double nu;
        double gamma;
        double D0;
        double Mweight;
        double Mvolume;
        bool bx,by,bz;
        Rand rng;
        vector<double> c;

        // cuda members
        int size;
        double * c_d;       			// concentration array
        double * df_d;      			// chemical potential array
        double * cpyBuff_d; 			// Copy buffer for ansynchronous data transfer
        double * chi_d;     			// interaction parameter
        double * Mob_d;     			// mobility
        double * nonUniformLap_d;	// laplacian of mobility and df
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
