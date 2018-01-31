
# include "CuesoExecute.h"

int main()
{

//	----------------------------------------------------------
//	Create object that executes MESO simulation:
//	----------------------------------------------------------

	CuesoExecute currentJob;

//	----------------------------------------------------------
//	Create simulation objects:
//	----------------------------------------------------------

	currentJob.createCuesoObjects();

//	----------------------------------------------------------
//	Execute the simulation:
//	----------------------------------------------------------

	currentJob.executeCuesoSimulation();

}
