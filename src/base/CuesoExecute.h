
# ifndef MESOEXECUTE_H
# define MESOEXECUTE_H

// ---------------------------------------------------------------------
// This is the Cueso executioner class.  This class runs the simulation.
// ---------------------------------------------------------------------

# include "CuesoBase.h"
# include <vector>
using namespace std;

class CuesoExecute {

   private:

   vector<CuesoBase*> cuesoapps;

   public:

      CuesoExecute();
      ~CuesoExecute();
      void createCuesoObjects();
      void executeCuesoSimulation();
      void runAppUnitTests();

};

# endif  // MESOEXECUTE_H
