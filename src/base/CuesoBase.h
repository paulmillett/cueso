
# ifndef CUESOBASE_H
# define CUESOBASE_H

# include "../utils/GetPot"
# include <string>

using std::string;

// ---------------------------------------------------------------------
// This is the base class for the Cueso project.  All application classes
// in the Cueso project inherent from this class.
// ---------------------------------------------------------------------

class CuesoBase {

    public:
        string appName;

        // -------------------------------------------------------------------
        // Define factory method that creates objects of CuesoBase sub-classes:
        // -------------------------------------------------------------------

        static CuesoBase* CuesoObjectFactory(string specifier);

        // -------------------------------------------------------------------
        // All sub-classes must define the below pure virtual functions:
        // -------------------------------------------------------------------

        virtual void initSystem() = 0;
        virtual void computeInterval(int) = 0;
        virtual void writeOutput(int) = 0;
        virtual void runUnitTests() = 0;

        // -------------------------------------------------------------------
        // Virtual destructor:
        // -------------------------------------------------------------------

        virtual ~CuesoBase()
        {
        }
};

# endif  // CUESOBASE_H
