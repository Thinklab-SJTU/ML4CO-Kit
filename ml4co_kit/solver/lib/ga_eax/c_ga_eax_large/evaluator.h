#ifndef __EVALUATOR__
#define __EVALUATOR__

#ifndef __INDI__
#include "indi.h"
#endif


#include <string.h>
#include <assert.h>

class TEvaluator {
 public:
  TEvaluator();
  ~TEvaluator();
  void SetInstance( char filename[] );
  void DoIt( TIndi& indi );
  int Direct( int i, int j );           // Large 
  void TranceLinkOrder( TIndi& indi );  // Large
  int GetOrd( int a, int b );           // Large

  void WriteTo( FILE* fp, TIndi& indi );    
  bool ReadFrom( FILE* fp, TIndi& indi );   
  bool CheckValid( int* array, int value ); 

  char fType[ 80 ];                      // Large 
  int fNearNumMax;
  int **fNearCity;
  int **fEdgeDisOrder;                   // Large
  int Ncity;
  double *x;
  double *y;
};
  

#endif



