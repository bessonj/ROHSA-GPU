#include "hypercube.h"
#include "model.h"
#include "algo_rohsa.h"
#include <omp.h>
#include <cmath>
#include <iostream>

int main()
{

	double temps1 = omp_get_wtime();
	double temps1_lecture = omp_get_wtime();

	model modeles_parametres;
        hypercube Hypercube_file(modeles_parametres); //utilise le fichier dat sans couper les données
//        hypercube Hypercube_file(modeles_parametres, 0,200); //Lecture des données, attentions aux indices en c++
//which part of the fits do you want to use ?  ind_debut, ind_fin
//set dimensions of the cube

////std::cout<<"aaa"+"bbb"<<std::end; //change filenames

//	Hypercube_file.display(Hypercube_file.data, 100); //affiche une tranche du tableau data à l'indice i=100
//	Hypercube_file.display_data(100); //affiche une tranche de data à l'indice 100
//	Hypercube_file.display_cube(100); //affiche une tranche de cube à l'indice 100
	
	double temps2_lecture = omp_get_wtime();


	algo_rohsa algo(modeles_parametres, Hypercube_file);


	Hypercube_file.plot_line(modeles_parametres.fit_params, 25, 12, modeles_parametres.n_gauss);

	Hypercube_file.display_result(modeles_parametres.fit_params, 100, modeles_parametres.n_gauss);
	Hypercube_file.display_cube(100); //affiche une tranche de cube à l'indice 100

	double temps2 = omp_get_wtime();
	std::cout<<"Temps de lecture : "<<temps2_lecture - temps1_lecture <<std::endl;
	std::cout<<"Temps total : "<<temps2 - temps1 <<std::endl; 

}






