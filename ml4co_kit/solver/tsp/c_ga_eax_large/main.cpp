#ifndef __ENVIRONMENT__
#include "env.h"
#endif

#include <stdio.h>
#include <stdlib.h>


int main( int argc, char* argv[] )
{
  int maxNumOfTrial;
  
  sscanf( argv[1], "%d", &maxNumOfTrial );
  char* dstFile = argv[2];
  
  TEnvironment* gEnv = NULL;
  gEnv = new TEnvironment();
  InitURandom();  

  int d;
  sscanf( argv[3], "%d", &d );
  gEnv->fNumOfPop = d;
  sscanf( argv[4], "%d", &d );
  gEnv->fNumOfKids = d;
  gEnv->fFileNameTSP = argv[5];
  gEnv->fFileNameInitPop = NULL;
  sscanf( argv[6], "%d", &d );
  gEnv->showInfo = d;
  if( argc == 8 )
    gEnv->fFileNameInitPop = argv[7];

  gEnv->Define();
  
  for( int n = 0; n < maxNumOfTrial; ++n )
  { 
    gEnv->DoIt();

    if (gEnv->showInfo){gEnv->PrintOn( n, dstFile );}       
    gEnv->WriteBest( dstFile );  
    // gEnv->WritePop( n, dstFile );
  }
  
  return 0;
}
