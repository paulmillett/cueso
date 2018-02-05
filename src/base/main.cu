
# include "CuesoExecute.h"
# include <iostream>
# include "../utils/GetPot"

using std::cout;
using std::endl;

int main()
{
    try
    {

        //	------------------------------------------------------
        //	Create object that executes MESO simulation:
        //	------------------------------------------------------

        CuesoExecute currentJob;

        //	------------------------------------------------------
        //	Run unit tests or simulation:
        //	------------------------------------------------------

        GetPot InParams("input.dat");
        bool runTests = InParams("UnitTests/runTests",0);

        if(runTests)
        {

            //	--------------------------------------------------
            //	Execute unit tests for each App:
            //	--------------------------------------------------

            currentJob.runAppUnitTests();

        }
        else
        {

            //	--------------------------------------------------
            //	Create simulation objects:
            //	--------------------------------------------------

            currentJob.createCuesoObjects();

            //	--------------------------------------------------
            //	Execute the simulation:
            //	--------------------------------------------------

            currentJob.executeCuesoSimulation();
        }
    }
    catch (const char* errMsg)
    {
        cout << errMsg << endl;
    }
}
