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
}
//	int nside = file.get_nside();
//	std::cout << " nside = " << nside <<std::endl;

//	file.multiresolution(nside); //Multirésolution pour nside = 5 (multiple de dim_v, la longueur native du FITS)	

/*	double temps1_m = omp_get_wtime(); 

	double temps1 = omp_get_wtime(); 

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
 
	return 0;
}
*/


// hypercube : entrée  sortie

// modèle, hypercube, algorithme

// boucle itérative évidente ==> algo

// template

//this->

//hypercube T : device ou host

// Envoyer un découpage en classe le 28 ou 29
