#include "hypercube.hpp"
#include "model.hpp"
#include "algo_rohsa.hpp"
#include "gradient.hpp"
#include <omp.h>
#include <cmath>
#include <iostream>

#include <string.h>

#define IND(x,y,z,g) N_g*N_z*N_y*x + N_g*N_z*y + N_g*z + g
#define IND(x,y,z) N_y*N_z*x+N_z*y+z

int main()
{
	//to truncate or not to truncate, that's the question
	bool whole_data_in_cube = true; //spatially

// test index tableaux
//-----------------------------------------------------------
	int taille_tab[] = {2,2,2};
	int product_taille_tab = taille_tab[0]*taille_tab[1]*taille_tab[2];
	size_t size_tab = product_taille_tab * sizeof(double);
	double* tab = (double*)malloc(size_tab);

	int N_x=2, N_y=2, N_z=2;

    tab[IND(1,1,1)]=2;

	std::cout<<tab[IND(1,1,1)]<<std::endl;
//-----------------------------------------------------------



	double temps1 = omp_get_wtime();
	double temps1_lecture = omp_get_wtime();

	model modeles_parametres;


	if(modeles_parametres.file_type_fits){

		//Pour un FITS :
        hypercube Hypercube_file(modeles_parametres, modeles_parametres.slice_index_min, modeles_parametres.slice_index_max, whole_data_in_cube); 

//		Hypercube_file.display_cube(0);
//
//		exit(0);
//Lecture des données, on regarde un extrait entre 2 indices
//set dimensions of the cube
//	Hypercube_file.display(Hypercube_file.data, 100); //affiche une tranche du tableau data à l'indice i=100
//	Hypercube_file.display_data(100); //affiche une tranche de data à l'indice 100
//	Hypercube_file.display_cube(100); //affiche une tranche de cube à l'indice 100
	
		double temps2_lecture = omp_get_wtime();

		algo_rohsa algo(modeles_parametres, Hypercube_file);

        double temps2 = omp_get_wtime();
        std::cout<<"Temps de lecture : "<<temps2_lecture - temps1_lecture <<std::endl;
        std::cout<<"Temps total (hors enregistrement): "<<temps2 - temps1 <<std::endl;


		int i_max = 5;
		int j_max = 5;

		for(int i=0;i<i_max;i++){
			for(int j=0;j<j_max;j++){
				if(i==j){
		Hypercube_file.plot_line(modeles_parametres.fit_params, i, j, modeles_parametres.n_gauss);
			}
			}
		}

	//	Hypercube_file.display_result(modeles_parametres.grid_params, 100, modeles_parametres.n_gauss); //affiche une tranche du cube reconstitué du modèle à l'indice 100
	//	Hypercube_file.display_cube(100); //affiche une tranche de cube à l'indice 100

		for(int p = 0; p<modeles_parametres.slice_index_max-modeles_parametres.slice_index_min-1; p++){
			Hypercube_file.display_result_and_data(modeles_parametres.grid_params, p, modeles_parametres.n_gauss, false); //affiche cote à cote les données et le modèle
		}
/*	for(int p = 0; p<modeles_parametres.slice_index_max-modeles_parametres.slice_index_min-1; p++){
		Hypercube_file.display_result_and_data(modeles_parametres.grid_params, p, modeles_parametres.n_gauss); //affiche cote à cote les données et le modèle
	}
*/

	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,0,0 , 1);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,0,1 , 2);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,0,2 , 3);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,1,0 , 4);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,1,1 , 5);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,1,2 , 6);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,2,0 , 7);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,2,1 , 8);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,2,2 , 9);

	}

	if(modeles_parametres.file_type_dat){

	//Pour un DAT :
        hypercube Hypercube_file(modeles_parametres, modeles_parametres.slice_index_min, modeles_parametres.slice_index_max, whole_data_in_cube); 
//	        hypercube Hypercube_file(modeles_parametres); 
	//utilise le fichier dat sans couper les données

//set dimensions of the cube

////std::cout<<"aaa"+"bbb"<<std::end; //change filenames

//	Hypercube_file.display(Hypercube_file.data, 100); //affiche une tranche du tableau data à l'indice i=100
//	Hypercube_file.display_data(100); //affiche une tranche de data à l'indice 100
//	Hypercube_file.display_cube(100); //affiche une tranche de cube à l'indice 100
	
//		Hypercube_file.display(Hypercube_file.cube,1);
//		Hypercube_file.display_cube(1);

//		exit(0);



		double temps2_lecture = omp_get_wtime();

		algo_rohsa algo(modeles_parametres, Hypercube_file);

        double temps2 = omp_get_wtime();
        std::cout<<"Temps de lecture : "<<temps2_lecture - temps1_lecture <<std::endl;
        std::cout<<"Temps total (hors enregistrement): "<<temps2 - temps1 <<std::endl;


		int i_max = 4;
		int j_max = 5;

		for(int i=0;i<i_max;i++){
			for(int j=0;j<j_max;j++){
				Hypercube_file.plot_line(modeles_parametres.fit_params, i, j, modeles_parametres.n_gauss);
			}
		}

//	Hypercube_file.display_result(modeles_parametres.grid_params, 100, modeles_parametres.n_gauss); //affiche une tranche du cube reconstitué du modèle à l'indice 100
//	Hypercube_file.display_cube(100); //affiche une tranche de cube à l'indice 100

		for(int p = 0; p<modeles_parametres.slice_index_max-modeles_parametres.slice_index_min-1; p++){
			Hypercube_file.display_result_and_data(modeles_parametres.grid_params, p, modeles_parametres.n_gauss, true); //affiche cote à cote les données et le modèle
		}


	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,0,0 , 1);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,0,1 , 2);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,0,2 , 3);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,1,0 , 4);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,1,1 , 5);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,1,2 , 6);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,2,0 , 7);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,2,1 , 8);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,2,2 , 9);

		}


	}






