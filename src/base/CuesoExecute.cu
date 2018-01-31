
# include "CuesoExecute.h"
# include "../utils/GetPot"

using std::vector;

// -------------------------------------------------------------------------
// Constructor:
// -------------------------------------------------------------------------

CuesoExecute::CuesoExecute()
{

    // -----------------------------------
    // make output directories:
    // -----------------------------------

    std::system("mkdir -p vtkoutput");            // make output directory
    std::system("exec rm -rf vtkoutput/*.vtk");   // remove any existing files

}



// -------------------------------------------------------------------------
// Destructor:
// -------------------------------------------------------------------------

CuesoExecute::~CuesoExecute()
{
}



// -------------------------------------------------------------------------
// Create the Cueso simulation objects, and store them in a vector:
// -------------------------------------------------------------------------

void CuesoExecute::createCuesoObjects()
{

    // ------------------------------------------------
    // 'GetPot' object containing input parameters:
    // ------------------------------------------------

    GetPot InParams("input.dat");

    // ------------------------------------------------
    // make vector of input section names:
    // ------------------------------------------------

    vector<string> sections = InParams.get_section_names();

    // ------------------------------------------------
    // determine which sections are executable 'apps':
    // ------------------------------------------------

    for (size_t i=0; i<sections.size(); i++) 
    {

        // ---------------------------------------------
        // get string that stores value of "section/app"
        // ---------------------------------------------

        string currentSec = sections[i] + "app";
        const char* currentSecChar = currentSec.c_str();
        string appFlag = InParams(currentSecChar,"false");

        // ---------------------------------------------
        // if "app = true", make a new object:
        // ---------------------------------------------

        if (appFlag == "true")
            cuesoapps.push_back(CuesoBase::CuesoObjectFactory(sections[i]));

    }

    // ------------------------------------------------
    // loop over executable objects, initializing each:
    // ------------------------------------------------

    for (size_t i=0; i<cuesoapps.size(); i++) 
    {
        cuesoapps[i]->initSystem();
        cuesoapps[i]->writeOutput(0);
    }

}



// -------------------------------------------------------------------------
// Execute the simulation by marching forward in time:
// -------------------------------------------------------------------------

void CuesoExecute::executeCuesoSimulation()
{

    // -----------------------------------
    // get the number of time steps:
    // -----------------------------------

    GetPot InParams("input.dat");
    int nstep = InParams("Time/nstep",0);
    int numOutputs = InParams("Output/numOutputs",1);

    // -----------------------------------
    // determine output interval:
    // -----------------------------------

    int outInterval = 0;
    if (numOutputs != 0) outInterval = nstep/numOutputs;
    if (numOutputs == 0) outInterval = nstep+1;

    // -----------------------------------
    // MARCH THROUGH TIME:
    // -----------------------------------

    for (int interval=1; interval<=numOutputs; interval++) {

        // --------------------------------
        // call 'StepForward' for each app:
        // --------------------------------

        for (size_t i=0; i<cuesoapps.size(); i++) 
            cuesoapps[i]->computeInterval(interval);

        // --------------------------------
        // write output for each app:
        // --------------------------------

        for (size_t i=0; i<cuesoapps.size(); i++) 
            cuesoapps[i]->writeOutput(outInterval*interval);
    }

}
