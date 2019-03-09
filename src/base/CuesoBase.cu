
# include "CuesoBase.h"

// -------------------------------------------------------------------------
// List of header files that need to be included...
// -------------------------------------------------------------------------

// Phase-Field classes:
# include "../phase_field/PFApp.h"
# include "../phase_field/PFSips.h"

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

   if (specifier == "PFApp/") 
   {
       CuesoBase* app = new PFApp(InParams);
       app->appName = "PFApp";
       return app;
   }
 	if (specifier == "PFSips/")
   {
   	CuesoBase* app = new PFSips(InParams);
   	app->appName = "PFSips";
   	return app;
   }

   // return null if specifier is not expected
   return nullptr;

}
