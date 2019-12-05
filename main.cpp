#include "hypercube.h"
#include "model.h"
#include "algo_rohsa.h"
#include <omp.h>

int main()
{
	double temps1 = omp_get_wtime();
	double temps1_lecture = omp_get_wtime();
        hypercube file("./GHIGLS.fits"); //Lecture des données
	model modeles_parametres;
	double temps2_lecture = omp_get_wtime();
	algo_rohsa algo(modeles_parametres, file);
	double temps2 = omp_get_wtime();
	std::cout<<"Temps de lecture : "<<temps2_lecture - temps1_lecture <<std::endl;
	std::cout<<"Temps total : "<<temps2 - temps1 <<std::endl; 
	exit(0);
}

