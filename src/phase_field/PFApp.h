
# ifndef PFAPP_H
# define PFAPP_H

# include <vector>
# include "../base/CuesoBase.h"
# include "../utils/rand.h"
# include "../utils/GetPot"

using std::vector;

class PFApp : public CuesoBase {

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
        Rand rng;
        vector<double> c;

        // cuda members
        int size;
        double * c_d;       // concentration array
        double * df_d;      // chemical potential array
        double * cpyBuff_d; // Copy buffer for ansynchronous data transfer
        double * rndm_d;    // random number for thermal fluctuation 
        dim3 blocks;
        dim3 blockSize;

    public:

        PFApp(const GetPot&);
        ~PFApp();
        void initSystem();
        void computeInterval(int interval);
        void writeOutput(int step);
        void runUnitTests();

    private:
        // unit tests
        bool lapKernUnitTest();

};

# endif  // PFAPP_H
