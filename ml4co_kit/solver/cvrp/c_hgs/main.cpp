#include "Genetic.h"
#include "commandline.h"
#include "LocalSearch.h"
#include "Split.h"
using namespace std;

int main(int argc, char *argv[])
{
	try
	{
		// Reading the arguments of the program
		CommandLine commandline(argc, argv);

		// Reading the data file and initializing some data structures
		if (commandline.show_info){
			std::cout << "----- READING DATA SET: " << commandline.pathInstance << std::endl;
		}
		Params params(commandline.pathInstance, commandline.nbVeh, commandline.seed, commandline.show_info);

		// Creating the Split and local search structures
		Split split(&params);
		LocalSearch localSearch(&params);

		// Initial population
		if (commandline.show_info){
			std::cout << "----- INSTANCE LOADED WITH " << params.nbClients << " CLIENTS AND " << params.nbVehicles << " VEHICLES" << std::endl;
			std::cout << "----- BUILDING INITIAL POPULATION" << std::endl;
		}
		Population population(&params, &split, &localSearch);

		// Genetic algorithm
		if (commandline.show_info){
			std::cout << "----- STARTING GENETIC ALGORITHM" << std::endl;
		}
		Genetic solver(&params, &split, &population, &localSearch);
		solver.run(commandline.nbIter, commandline.timeLimit, commandline.show_info);
		if (commandline.show_info){
			std::cout << "----- GENETIC ALGORITHM FINISHED, TIME SPENT: " << (double)clock()/(double)CLOCKS_PER_SEC << std::endl;
		}

		// Exporting the best solution
		if (population.getBestFound() != NULL)
		{
			population.getBestFound()->exportCVRPLibFormat(commandline.pathSolution, commandline.show_info);
			population.exportSearchProgress(commandline.pathSolution + ".PG.csv", commandline.pathInstance, commandline.seed);
			if (commandline.pathBKS != "") population.exportBKS(commandline.pathBKS, commandline.show_info);
		}
	}
	catch (const string& e) { std::cout << "EXCEPTION | " << e << std::endl; }
	catch (const std::exception& e) { std::cout << "EXCEPTION | " << e.what() << std::endl; }
	return 0;
}
