#include <iostream>
#include <getopt.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <assert.h>
#include <limits>

using namespace std;


///////////////////////////////////////////////////////
/* VARIABLES of the generator */

int n = 1000;  // Number of nodes
int m = 3700;  // Number of clauses
int k = 0;    // Average size of clauses (flexible part)
int K = 3;    // Rigid clause size
double b = 0.83; // Beta for vars
double B = 1.00; // Beta for clauses
double T = 1.9; // Temperature
bool varRename = false;  // if true varRename variables acording to angle
int seed = 0; // Random seed
bool pgraph = false;

vector<vector<double> > radius; // Vector representing the radius in [0,n) for variables and in [0,m) for clauses
// radius[0] for variables and radius[1] for clauses
vector <vector<double> > angle; // Vector representing the angle in [0,2PI)
// angle[0] for variables and angle[1] for clauses
vector<vector<int> > neighs;    // Vector representing neighbors of a clause node
int edges = 0;                  // Edges already created


///////////////////////////////////////////////////////
/* Print the usage of this program */
void printUsage(char* prog){
  cerr << "c Popularity-Similarity SAT Instance Generator" << endl;
  cerr << "c Created by Jesús Giráldez Crú and Jordi Levy" << endl;
  cerr << "c" << endl;
  cerr << "c Usage: " << prog << " [arguments]" << endl;
  cerr << "c    Arguments:" << endl;
  cerr << "c         -n <int>   : number of nodes (default=100)" << endl;
  cerr << "c         -m <int>   : number of clauses (default=400)" << endl;
  cerr << "c         -k <int>   : average clause size (default=0)" << endl;
  cerr << "c         -K <int>   : clause size for regular model (default=0)" << endl;
  cerr << "c         -b <float> : beta for variables (default=1)" << endl;
  cerr << "c         -B <float> : beta for clauses (default=1)" << endl;
  cerr << "c         -T <float> : temperature (default=0)" << endl;
  cerr << "c         -r         : varRename variables and reorder clauses so similar ones are closer (default=false)" << endl;
  cerr << "c         -s <int>   : random seed (default=0)" << endl;
  cerr << "c         -g         : print graph instead of CNF (default=false)" << endl;
  cerr << "c" << endl;
}

///////////////////////////////////////////////////////
/* Print the usage of this program and exit */
void printUsageAndExit(char* prog, int code){
  printUsage(prog);
  exit(code);
}

///////////////////////////////////////////////////////
/* Parse the arguments given to the program */
void parseArgs(int argc, char** argv){
  int opt;
  // Parse argument
  while((opt=getopt(argc, argv, "n:m:k:K:b:B:T:s:rgh?")) != -1){
    switch(opt){
      case 'n':
      n = atoi(optarg);
      break;
      case 'm':
      m = atoi(optarg);
      break;
      case 'k':
      k = atoi(optarg);
      break;
      case 'K':
      K = atoi(optarg);
      break;
      case 'b':
      b = atof(optarg);
      break;
      case 'B':
      B = atof(optarg);
      break;
      case 'T':
      T = atof(optarg);
      break;
      case 's':
      seed = atoi(optarg);
      break;
      case 'r':
      varRename = true;
      break;
      case 'g':
      pgraph = true;
      break;
      case 'h':
      printUsageAndExit(argv[0], 0);
      break;
      case '?':
      printUsageAndExit(argv[0], 0);
      break;
      default:
      cerr << "c ERROR: unrecognised argument" << endl;
      printUsageAndExit(argv[0], -1);
    }
  }

  // Checks argument values
  // number of nodes : n
  if(n<1){
    cerr << "ERROR: n (number of nodes) must be greater than 0" << endl;
    printUsageAndExit(argv[0], -1);
  }
  // number of connections : m
  if(m<1){
    cerr << "ERROR: m (number of clauses) must be greater than 0" << endl;
    printUsageAndExit(argv[0], -1);
  }
  // temperature : T
  if(T < 0){
    cerr << "ERROR: T (temperature) must be greater or equal than 0" << endl;
    printUsageAndExit(argv[0], -1);
  }
  // beta : b
  if(b < 0 || b > 1 || B < 0 || B > 1){
    cerr << "ERROR: b and B (beta) must be in the interval [0,1]" << endl;
    printUsageAndExit(argv[0], -1);
  }
}

///////////////////////////////////////////////////////
/* Given a vector x, returns the order of its elements, or the inverse */
bool myorder(pair<int,double>x, pair<int,double>y) {
  return x.second < y.second;
}
void getorder(vector <double> x, vector <int> &y, bool inverse){
  vector <pair<int,double> >z(x.size());
  y.resize(x.size());
  for(int i=0; i<x.size(); i++) {
    z[i].first = i;
    z[i].second = x[i];
  }
  sort(z.begin(), z.end(), myorder);
  if (inverse) {
    for(int i=0; i<x.size(); i++)
    y[z[i].first] = i;
  } else {
    for(int i=0; i<x.size(); i++)
    y[i] = z[i].first;
  }
}

///////////////////////////////////////////////////////
/* Returns true if there exists an edge between variable nodes i and clause node j */
bool checkEdge(int i, int j){
  return find(neighs[j].begin(), neighs[j].end(), i) != neighs[j].end();
}

///////////////////////////////////////////////////////
/* Create an edge between variable i and clause j (or vice verse if isvar=false) */
void createEdge(int i, int j){
  //cerr << "c New edge "<<i<<" - "<<j<<endl;
  assert(i>0 && i<=n && j>0 && j<=m);
  assert(!checkEdge(i,j));
  edges++;
  neighs[j].push_back(i);
}

///////////////////////////////////////////////////////
/* Print the resulting graph */
void printGraph(){
  double minRadVar = 0, minRadClau = 0;
  for(int i=1; i<=n; i++)if(log(radius[0][i]) < minRadVar) minRadVar = log(radius[0][i]);
  for(int j=1; j<=m; j++) if(log(radius[1][j]) < minRadClau) minRadClau = log(radius[1][j]);
  double size = 1;
  cout << "graph{size=\""<< 2*size << "," << 2*size << "\";\nnode [style=filled,width=0.05,height=0.05];\n";
  for(int i=1; i<=n; i++)
    cout << "n" << i << " [color=red shape=\"ellipse\" pos=\"" << 10*size*(log(radius[0][i])-minRadVar)*cos(angle[0][i]) << "," << 10*size*(log(radius[0][i])-minRadVar)*sin(angle[0][i]) << "!\"]\n";
  for(int j=1; j<=m; j++)
    cout << "c" << j << " [color=blue shape=\"box\" pos=\"" << 10*size*(log(radius[1][j])-minRadClau)*cos(angle[1][j]) << "," << 10*size*(log(radius[1][j])-minRadClau)*sin(angle[1][j]) << "!\"]\n";
  for(int j=1; j<neighs.size(); j++){
    cout << "c" << j << " -- { ";
    for(int i=0; i<neighs[j].size(); i++)
      cout << "n" << neighs[j][i] << " ";
    cout << "};\n";
  }
  cout << "}\n";
}

///////////////////////////////////////////////////////
/* Print the resulting SAT instance (including sign for literals) */
void printFormula(){
  cout << "c Popularity-Similarity SAT Instance Generator" << endl;
  cout << "c Created by Jesús Giráldez Crú and Jordi Levy" << endl;
  cout << "c" << endl;
  cout << "c #variable     n = " << n << endl;
  cout << "c #clauses      m = " << m << endl;
  cout << "c minClauseSize K = " << K << endl;
  cout << "c avgClauseSize k = " << k << endl;
  cout << "c betaVar       b = " << b << endl;
  cout << "c betaClau      B = " << B << endl;
  cout << "c temperature   T = " << T << endl;
  cout << "c #edges     size = " << edges << endl;
  cout << "c" << endl;
  cout << "p cnf " << n << " "<< neighs.size()-1 << endl;

  if (!varRename) {
    vector <int>indvar, indcla;

    getorder(radius[0],indvar,true);
    getorder(radius[1],indcla,false);

    for(int j=1; j<neighs.size(); j++){
      k = indcla[j];
      if(neighs[k].size() > 0){
        for(int i=0; i<neighs[k].size(); i++)
        cout << indvar[neighs[k][i]] * (rand()%2==0?1:-1) << " ";
        cout << "0" << endl;
      }
      else
      cerr << "Warning: generated empty clause" << endl;
    }
  } else {
    vector <int>indvar, indcla;

    getorder(angle[0],indvar,true);
    getorder(angle[1],indcla,false);

    for(int j=1; j<neighs.size(); j++){
      k = indcla[j];
      if(neighs[k].size() > 0){
        for(int i=0; i<neighs[k].size(); i++)
        cout << indvar[neighs[k][i]] * (rand()%2==0?1:-1) << " ";
        cout << "0" << endl;
      }
      else
      cerr << "Warning: generated empty clause" << endl;
    }
  }
}

///////////////////////////////////////////////////////
/* Insert an element in an ordered list, and keep ordered */
template<typename value>
void insertOrdered(vector<pair<value,double> >& l, pair<value,double> elem){
  int pos;

  for(pos=0; pos<l.size(); pos++){
    if(elem.second < l[pos].second){
      break;
    }
  }
  l.resize(l.size()+1);

  for(int i=l.size()-1; i>pos; i--)
  l[i] = l[i-1];
  l[pos] = elem;
}

///////////////////////////////////////////////////////
double myabs(double k){
  return k >= 0 ? k : -k;
}

///////////////////////////////////////////////////////
/* Radial coordinates in PS model */
double radiusPS(){
  return ((double)rand() / (double)RAND_MAX);
}

///////////////////////////////////////////////////////
/* Angular coordinate in PS model */
double anglePS(){
  return ((double)rand() / (double)RAND_MAX) * 2 * M_PI;
}

///////////////////////////////////////////////////////
/* Computes hyperbolic distance between variable i and clause j */
double hyperDist(int i, int j) {
  if (checkEdge(i,j)) return numeric_limits<double>::max();
  double diffang = M_PI - myabs(M_PI - myabs(angle[1][j] - angle[0][i]));
  double ri = radius[0][i];
  double rj = radius[1][j];
  return pow(ri,b) * pow(rj,B) * diffang;
}



///////////////////////////////////////////////////////

int main(int argc, char** argv){

  int counter[2][2] = {{0,0},{0,0}};

  // Parse arguments
  parseArgs(argc, argv);

  // Initialize
  srand(seed);
  neighs.resize(m+1);
  angle.resize(2);
  angle[0].resize(n+1);
  angle[1].resize(m+1);

  radius.resize(2);
  radius[0].resize(n+1);
  radius[1].resize(m+1);

  // Compute angle location
  for(int i=1; i<=n; i++){
    angle[0][i] = anglePS();
  }
  for(int i=1; i<=m; i++){
    angle[1][i] = anglePS();
  }

  // Compute radius location
  for(int i=1; i<=n; i++){
    radius[0][i] = radiusPS();
  }
  for(int i=1; i<=m; i++){
    radius[1][i] = radiusPS();
  }

  // Add edges for fixed arity (K>0)
  if (K>0) {
    if (T==0) {
      for (int j=1; j<=m; j++) {
        vector<pair<int,double> > selected;
        for (int i=1; i<=n; i++) {
          if (selected.size() < K)
          insertOrdered(selected, make_pair(i,hyperDist(i,j)));
          else {
            double d = hyperDist(i,j);
            if(d < selected[selected.size()-1].second){
              insertOrdered(selected, make_pair(i,d));
              selected.pop_back();
            }
          }
        }
        for(int i=0; i<selected.size(); i++)
        createEdge(selected[i].first, j);
      }

    } else { // T>0
      for (int j=1; j<=m; j++) {
        int e=0, eold;
        double SP=0, SPold;

        // Computes a first approximation for sum of probabilities
        for (int i=1; i<=n; i++)
        SP += 1.0 / pow(hyperDist(i,j), 1.0/T);

        // Iterates re-calculation of SP until no probability needs being truncated
        do {
          SPold = SP;
          eold = e;
          SP=0;
          for (int i=1; i<=n; i++) {
            double prob = 1.0 / pow(hyperDist(i,j), 1.0/T);
            if (prob  * (K - eold) / SPold >= 1)  // If prob needs being truncated, generate corresponding edge
            { createEdge(i,j); e++; counter[0][0]++; }
            else   // else consider probability for next phase
            SP += prob;
          }
        } while (eold < e); //SP=SPOld and e=eold, therefore finish iteration

        // Starts random generation of edges
        //for (int i=1; i<=n; i++) {
        while (e < K) {
          int i = rand()%n+1;
          double prob = 1.0 / pow(hyperDist(i,j), 1.0/T) * (K - eold) / SP;
          if( !checkEdge(i,j) && ((double)rand() / (double)RAND_MAX) < prob)
          { createEdge(i,j); e++;  counter[0][1]++; }
        }
      }
      cerr << "Rigid edges: " << counter[0][0] << " (truncated) + " << counter[0][1] << endl;
    }
  }

  // Add edges for flexible arity (k>0)
  if (k>0) {
    if (T==0) {
      vector<pair<pair<int,int>,double> > selected;
      for (int i=1; i<=n; i++)
      for (int j=1; j<=m; j++) {
        if (!checkEdge(i,j)) {
          if (selected.size() < k*m)
          insertOrdered(selected, make_pair(make_pair(i,j), hyperDist(i,j)));
          else {
            double d = hyperDist(i,j);
            if(d < selected[selected.size()-1].second){
              insertOrdered(selected, make_pair(make_pair(i,j),d));
              selected.pop_back();
            }
          }
        }
      }

      for(int i=0; i<selected.size(); i++)
      createEdge(selected[i].first.first, selected[i].first.second);

    } else { // T > 0
      int e=0, eold;
      double SP=0, SPold;

      // Computes a first approximation for sum of probabilities
      for (int i=1; i<=n; i++)
      for (int j=1; j<=m; j++)
      SP += 1.0 / pow(hyperDist(i,j), 1.0/T);

      // Iterates re-calculation of SP until no probability needs being truncated
      do {
        SPold = SP;
        eold = e;
        SP=0;
        for (int i=1; i<=n; i++)
        for (int j=1; j<=m; j++) {
          if (!checkEdge(i,j)) {
            double prob = 1.0 / pow(hyperDist(i,j), 1.0/T);
            if (prob  * (k * m - eold) / SPold >= 1)  // If prob needs being truncated, generate corresponding edge
            { createEdge(i,j); e++;  counter[1][0]++;}
            else   // else consider probability for next phase
            SP += prob;
          }
        }
        //cerr << "SP=" <<SP<<" e="<<e<<endl;
      } while (eold < e); //SP=SPOld and e=eold, therefore finish iteration
      cerr << "Flexible edges: " << counter[1][0] << " (truncated)";

      // Starts random generation of edges
      //for (int i=1; i<=n; i++)
      //for (int j=1; j<=m; j++) {
      while (e < k * m) {
        int i=rand()%n+1;
        int j=rand()%m+1;
        double prob = 1.0 / pow(hyperDist(i,j), 1.0/T) * (k * m - eold) / SP;
        if( !checkEdge(i,j) && ((double)rand() / (double)RAND_MAX) < prob)
        { createEdge(i,j); e++; counter[1][1]++; }
      }
      cerr << " + " << counter[1][1] << endl;
    }
  }

  // Finally print (cnf or graph)
  if(pgraph){
    printGraph();
  }else{
    printFormula();
  }

}



///////////////////////////////////////////////////////