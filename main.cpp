#include "Parse.h"
#include "Gaussian.h"
#include <omp.h>

int main()
{

        Parse file; //Lecture des données

//	int nside = file.get_nside();
//	std::cout << " nside = " << nside <<std::endl;

//	file.multiresolution(nside); //Multirésolution pour nside = 5 (multiple de dim_v, la longueur native du FITS)	

	double temps1_m = omp_get_wtime(); 

	for(int n_size(1); n_size <=32; n_size++)
	{
		file.multiresolution(n_size);
	}

	double temps1 = omp_get_wtime(); 

	Gaussian decomposition(file);

	double temps2 = omp_get_wtime(); 

	std::cout<<"Temps d'exécution décomposition :"<<temps2-temps1<<" secondes"<<std::endl;
	std::cout<<"Temps d'exécution multirésolution :"<<temps1-temps1_m<<" secondes"<<std::endl;

	double S=0.;
	for(int i(1);i<=32;i++)
	{
		S+=pow(i,2);
	}
	std::cout<<"nside**2/(temps2-temps1) ="<< S/(temps1-temps1_m)<<std::endl;

// rajouter vitesse calcul : nside**2/(temps2-temps1)  ==> pts/s
// le futur du auto_ptr
// makefile
// nom du GHIGLS
// profiler : Valgrind, kyle
// 
// multirésolution ht pmn boucle sur cube nmp (merci fortran) 

// combien on gagne sur le total
// gains 
// 15h30 : 
	return 0;
}

