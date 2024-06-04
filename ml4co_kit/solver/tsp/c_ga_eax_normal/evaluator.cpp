#ifndef __EVALUATOR__
#include "evaluator.h"
#endif

TEvaluator::TEvaluator()
{
  fEdgeDis = NULL;
  fNearCity = NULL;
  Ncity = 0;
  fNearNumMax = 50;  
}

TEvaluator::~TEvaluator()
{
  for ( int i = 0; i < Ncity; ++i ) 
    delete[] fEdgeDis[ i ];
  delete[] fEdgeDis;
  for ( int i = 0; i < Ncity; ++i ) 
    delete[] fNearCity[ i ];
  delete[] fNearCity;
  
  delete [] x;
  delete [] y;
}

void TEvaluator::SetInstance( char filename[] )
{
  FILE* fp;
  int n;
  int *check;
  char word[ 80 ], type[ 80 ];

  fp = fopen( filename, "r" );


  ////// read instance //////
  while( 1 ){
    if( fscanf( fp, "%s", word ) == EOF )
      break;

    if( strcmp( word, "DIMENSION" ) == 0 ){
      fscanf( fp, "%s", word ); 
      assert( strcmp( word, ":" ) == 0 );
      fscanf( fp, "%d", &Ncity ); 
    } 

    if( strcmp( word, "EDGE_WEIGHT_TYPE" ) == 0 ){
      fscanf( fp, "%s", word ); 
      assert( strcmp( word, ":" ) == 0 );
      fscanf( fp, "%s", type ); 
    } 

    if( strcmp( word, "NODE_COORD_SECTION" ) == 0 ) 
      break;
      

  }
  if( strcmp( word, "NODE_COORD_SECTION" ) != 0 ){
    printf( "Error in reading the instance\n" );
    exit(0);
  }

  x = new double [ Ncity ]; 
  y = new double [ Ncity ]; 
  int checkedN[ Ncity ];

  int xi, yi; 
  for( int i = 0; i < Ncity; ++i ) 
  {
    fscanf( fp, "%d", &n );
    assert( i+1 == n ); 
    fscanf( fp, "%s", word ); 
    x[ i ] = atof( word );
    fscanf( fp, "%s", word ); 
    y[ i ] = atof( word );
  }

  fclose(fp);
  //////////////////////////

  fEdgeDis = new int* [ Ncity ];
  for( int i = 0; i < Ncity; ++i ) fEdgeDis[ i ] = new int [ Ncity ];
  fNearCity = new int* [ Ncity ];
  for( int i = 0; i < Ncity; ++i ) fNearCity[ i ] = new int [ fNearNumMax+1 ];

  /////////////// Remove duplicated nodes /////////////
  /*
  check = new int [ Ncity ];
  for( int i = 0; i < Ncity; ++i )
    check[ i ] = 0;
  
  for( int i = 0; i < Ncity ; ++i )
  {
    for( int j = 0; j < i ; ++j )
    {
      fEdgeDis[i][j]=(int)(sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))+0.5);
      if( x[i] == x[j] && y[i] == y[j] )
	check[ i ] = 1;
    }
  }

  n = 0;
  for( int i = 0; i < Ncity ; ++i ){
    if( check[ i ] == 0 ){
      x[ n ] = x[ i ];
      y[ n ] = y[ i ];
      ++n;
    }
  }
  if( Ncity != n ){
    printf( "Ncity is changed %d to %d\n", Ncity, n ); 
    Ncity = n;
  }
  delete [] check;
  */
  ////////////////////////////////////////////////////


  if( strcmp( type, "EUC_2D" ) == 0  ) {
    for( int i = 0; i < Ncity ; ++i )
    {
      for( int j = 0; j < Ncity ; ++j )
      {
	fEdgeDis[ i ][ j ]=(int)(sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))+0.5);
      }
    }
  }
  else if( strcmp( type, "ATT" ) == 0  ) { 
    for( int i = 0; i < Ncity; ++i )
    {
      for( int j = 0; j < Ncity; ++j ) 
      {
	double r = (sqrt(((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]))/10.0));
	int t = (int)r;
	if( (double)t < r ) fEdgeDis[ i ][ j ] = t+1;
	else fEdgeDis[ i ][ j ] = t; 
      }
    }
  }
  else if( strcmp( type, "CEIL_2D" ) == 0  ) {  
    for( int i = 0; i < Ncity ; ++i )
    {
      for( int j = 0; j < Ncity ; ++j )
      {
	fEdgeDis[ i ][ j ]=(int)ceil(sqrt((x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j])));
      }
    }
  }
  else{
    printf( "EDGE_WEIGHT_TYPE is not supported\n" );
    exit( 1 );
  }

  int ci;
  int j1 ,j2 ,j3;
  int city_num = 0;
  int min_dis;

  for( ci = 0; ci < Ncity; ++ci )
  {
    for( j3 = 0; j3 < Ncity; ++j3 ) checkedN[ j3 ] = 0;
    checkedN[ ci ] = 1;
    fNearCity[ ci ][ 0 ] = ci;
    for( j1 = 1; j1 <= fNearNumMax; ++j1 ) 
    {
      min_dis = 100000000;
      for( j2 = 0; j2 < Ncity; ++j2 )
      { 
	if( fEdgeDis[ ci ][ j2 ] <= min_dis && checkedN[ j2 ] == 0 )
	{
	  city_num = j2;
          min_dis = fEdgeDis[ ci ][ j2 ];
	}
      }
      fNearCity[ ci ][ j1 ] = city_num;
      checkedN[ city_num ] = 1;
    }
  }
}


void TEvaluator::DoIt( TIndi& indi )
{
  int d;
  d = 0;  
  for( int i = 0; i < Ncity; ++i )
  {  
    d = d + fEdgeDis[ i ][ indi.fLink[i][0] ];
    d = d + fEdgeDis[ i ][ indi.fLink[i][1] ];
  }
  indi.fEvaluationValue = d/2;
}


void TEvaluator::WriteTo( FILE* fp, TIndi& indi ) 
{
  assert( Ncity == indi.fN );
  int Array[ Ncity ];
  int curr, next, pre, st, count;

  count = 0;
  pre = -1;
  curr = 0;
  st = 0;
  while( 1 )
  {
    Array[ count++ ] = curr + 1;

    if( count > Ncity ){
      printf( "Invalid\n" );
      return;
    } 
 
    if( indi.fLink[ curr ][ 0 ] == pre )
      next = indi.fLink[ curr ][ 1 ];
    else 
      next = indi.fLink[ curr ][ 0 ];

    pre = curr;
    curr = next;
    if( curr == st )
      break;
  }

  if( this->CheckValid( Array, indi.fEvaluationValue ) == false ){
    printf( "Individual is invalid \n" );
  }

  fprintf( fp, "%d %d\n", indi.fN, indi.fEvaluationValue );
  for( int i = 0; i < indi.fN; ++i )
    fprintf( fp, "%d ", Array[ i ] );
  fprintf( fp, "\n" ); 
}


bool TEvaluator::ReadFrom( FILE* fp, TIndi& indi )
{
  assert( Ncity == indi.fN );
  int Array[ Ncity ];
  int curr, next, pre, st, count;
  int N, value;

  if( fscanf( fp, "%d %d", &N, &value ) == EOF ) 
    return false;
  assert( N == Ncity );
  indi.fN = N;
  indi.fEvaluationValue = value;

  for( int i = 0; i < Ncity; ++i ){ 
    if( fscanf( fp, "%d", &Array[ i ] ) == EOF )
      return false;
  }

  if( this->CheckValid( Array, indi.fEvaluationValue ) == false ){
    printf( "Individual is invalid \n" );
    return false;
  }

  for( int i = 0; i < Ncity; ++i ){ 
    Array[ i ] -= 1; 
  }

  for( int i = 1; i < Ncity-1; ++i ){ 
    indi.fLink[ Array[ i ] ][ 0 ] = Array[ i-1 ]; 
    indi.fLink[ Array[ i ] ][ 1 ] = Array[ i+1 ]; 
  }
  indi.fLink[ Array[ 0 ] ][ 0 ] = Array[ Ncity-1 ]; 
  indi.fLink[ Array[ 0 ] ][ 1 ] = Array[ 1 ]; 
  indi.fLink[ Array[ Ncity-1 ] ][ 0 ] = Array[ Ncity-2 ]; 
  indi.fLink[ Array[ Ncity-1 ] ][ 1 ] = Array[ 0 ]; 

  return true;
}    


bool TEvaluator::CheckValid( int* array, int value )
{
  int check[ Ncity ];
  int distance;

  for( int i = 0; i < Ncity; ++i ){
    check[ i ] = 0;
  }

  for( int i = 0; i < Ncity; ++i )
    ++check[ array[ i ]-1 ];

  for( int i = 0; i < Ncity; ++i ){
    if( check[ i ] != 1 ){
      return false;
    }
  }
    
  distance = 0;  
  for( int i = 0; i < Ncity-1; ++i ){
    distance += fEdgeDis[ array[ i ]-1 ][ array[ i+1 ]-1 ];
  }
  distance += fEdgeDis[ array[ Ncity-1 ]-1 ][ array[ 0 ]-1 ];
  if( distance != value ){
    return false;
  }
  return true;
}
