
# include "CuesoBase.h"

// -------------------------------------------------------------------------
// List of header files that need to be included...
// -------------------------------------------------------------------------

// Phase-Field classes:
# include "../phase_field/PFApp.h"

// -------------------------------------------------------------------------
// Factory method: this function returns an object determined
// by the string 'specifier':
// {Note: all of the returnable objects inherent from 'CuesoBase'}
// -------------------------------------------------------------------------

CuesoBase* CuesoBase::CuesoObjectFactory(string specifier)
{

   // ------------------------------------------------
   // 'GetPot' object containing input parameters:
   // ------------------------------------------------

   GetPot InParams("input.dat");

   // -----------------------------------
   // return the requested object:
   // -----------------------------------

   if (specifier == "PFApp/") return new PFApp(InParams);

   // return null if specifier is not expected
   return nullptr;

}
