#include "hypercube.h"
#include "model.h"
#include "algo_rohsa.h"
#include <omp.h>
#include <cmath>
#include <iostream>

#include <string.h>

int main()
{
	double temps1 = omp_get_wtime();
	double temps1_lecture = omp_get_wtime();

	model modeles_parametres;

//Pour un FITS :
//        hypercube Hypercube_file(modeles_parametres, 150,350); //Lecture des données, on regarde un extrait entre 2 indices

//Pour un DAT :
        hypercube Hypercube_file(modeles_parametres); //utilise le fichier dat sans couper les données


//set dimensions of the cube

////std::cout<<"aaa"+"bbb"<<std::end; //change filenames

//	Hypercube_file.display(Hypercube_file.data, 100); //affiche une tranche du tableau data à l'indice i=100
//	Hypercube_file.display_data(100); //affiche une tranche de data à l'indice 100
//	Hypercube_file.display_cube(100); //affiche une tranche de cube à l'indice 100
	
	double temps2_lecture = omp_get_wtime();

	algo_rohsa algo(modeles_parametres, Hypercube_file);

	Hypercube_file.plot_line(modeles_parametres.fit_params, 25, 12, modeles_parametres.n_gauss);

	Hypercube_file.display_result(modeles_parametres.grid_params, 100, modeles_parametres.n_gauss); //affiche une tranche du cube reconstitué du modèle à l'indice 100
	Hypercube_file.display_cube(100); //affiche une tranche de cube à l'indice 100
	for(int p = 100; p<150; p++){
		Hypercube_file.display_result_and_data(modeles_parametres.grid_params, p, modeles_parametres.n_gauss); //affiche cote à cote les données et le modèle
	}


	double temps2 = omp_get_wtime();
	std::cout<<"Temps de lecture : "<<temps2_lecture - temps1_lecture <<std::endl;
	std::cout<<"Temps total : "<<temps2 - temps1 <<std::endl; 

}






