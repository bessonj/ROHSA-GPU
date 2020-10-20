#include "hypercube.hpp"
#include "parameters.hpp"
#include "algo_rohsa.hpp"
//#include "gradient.hpp"
#include <omp.h>
#include <cmath>
#include <iostream>

#include <string.h>

/*
#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE)[1]*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE)[2]*(t##_SHAPE)[1]*x+(t##_SHAPE)[2]*y+z]
*/

#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE1)*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE2)*(t##_SHAPE1)*x+(t##_SHAPE2)*y+z]

int main(int argc, char * argv[])
{

	//to truncate or not to truncate, that's the question
	bool whole_data_in_cube = true; //spatially
// test index tableaux
//-----------------------------------------------------------
	int tableau2D_SHAPE[] = {2,2};
	int tableau2D_SHAPE0 = 2;
	int tableau2D_SHAPE1 = 2;
	int tableau2D_SHAPE_number = tableau2D_SHAPE[0]*tableau2D_SHAPE[1];
	size_t tableau2D_size = tableau2D_SHAPE_number * sizeof(double);
	double* tableau2D = (double*)malloc(tableau2D_size);

	int tableau3D_SHAPE[] = {2,2,2};
	int tableau3D_SHAPE0 = 2;
	int tableau3D_SHAPE1 = 2;
	int tableau3D_SHAPE2 = 2;
	int tableau3D_SHAPE_number = tableau3D_SHAPE[0]*tableau3D_SHAPE[1];
	size_t tableau3D_size = tableau3D_SHAPE_number * sizeof(double);
	double* tableau3D = (double*)malloc(tableau3D_size);

	tableau2D[3]=2;
	tableau3D[1+2*1+0]=2;

	std::cout<<INDEXING_2D(tableau2D, 1, 1)<<std::endl;
	std::cout<<INDEXING_3D(tableau3D, 0, 1, 1)<<std::endl;

//-----------------------------------------------------------

	double temps1 = omp_get_wtime();
	double temps1_lecture = omp_get_wtime();

	parameters modeles_parametres(argv[1]);


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

		Hypercube_file.mean_parameters(modeles_parametres.grid_params, modeles_parametres.n_gauss);



/*
			for(int i=0;i<std::min(Hypercube_file.data[0].size(),Hypercube_file.data.size());i++){
				for(int j=0;j<std::min(Hypercube_file.data[0].size(),Hypercube_file.data.size());j++){
					if(i==j){
			Hypercube_file.plot_line(modeles_parametres.grid_params, i, j, modeles_parametres.n_gauss);
				}
				}
			}

		//	Hypercube_file.display_result(modeles_parametres.grid_params, 100, modeles_parametres.n_gauss); //affiche une tranche du cube reconstitué du modèle à l'indice 100
		//	Hypercube_file.display_cube(100); //affiche une tranche de cube à l'indice 100

			for(int p = 0; p<modeles_parametres.slice_index_max-modeles_parametres.slice_index_min-1; p++){
				Hypercube_file.display_result_and_data(modeles_parametres.grid_params, p, modeles_parametres.n_gauss, false); //affiche cote à cote les données et le modèle
			}

		//	for(int p = 40; p<70; p++){
		//		Hypercube_file.display_2_gaussiennes(modeles_parametres.grid_params, p, 0*3+1, 0, 1);
		//	}

			for(int num_gauss = 0; num_gauss < modeles_parametres.n_gauss; num_gauss ++){
				for (int num_par = 0; num_par < 3; num_par++)
				{
					Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params, num_gauss, num_par , num_par+num_gauss*3);
				}
			}
*/
exit(0);
//

	for(int p = 0; p<modeles_parametres.slice_index_max-modeles_parametres.slice_index_min-1; p++){

		for (int num_par = 0; num_par < 3; num_par++)
			{
			for(int num_gauss = 0; num_gauss < modeles_parametres.n_gauss; num_gauss ++)
				{
				for(int num_gauss_cur = 0; num_gauss_cur < modeles_parametres.n_gauss; num_gauss_cur ++)
					{
					if(num_gauss!=num_gauss_cur){
						Hypercube_file.display_2_gaussiennes_par_par_par(modeles_parametres.grid_params, p, num_gauss*3+num_par, 3*num_gauss+num_par, 3*num_gauss_cur+num_par);
					}
				}
			}
		}
	}
		exit(0);
//for(int deuxieme_gauss=0; deuxieme_gauss<)
	for(int p = 0; p<Hypercube_file.cube[0][0].size(); p++){
		for(int num_gauss = 0; num_gauss < modeles_parametres.n_gauss; num_gauss ++){
			for (int num_par = 0; num_par < 3; num_par++)
			{
				Hypercube_file.display_2_gaussiennes(modeles_parametres.grid_params, p, num_gauss*3+num_par, 0, 1);
			}
		}

		for(int num_gauss = 0; num_gauss < modeles_parametres.n_gauss; num_gauss ++){
			for (int num_par = 0; num_par < 3; num_par++)
			{
				Hypercube_file.display_2_gaussiennes(modeles_parametres.grid_params,p , num_gauss*3+num_par, 1, 2);
			}
		}

			for(int num_gauss = 0; num_gauss < modeles_parametres.n_gauss; num_gauss ++){
			for (int num_par = 0; num_par < 3; num_par++)
			{
				Hypercube_file.display_2_gaussiennes(modeles_parametres.grid_params, p, num_gauss*3+num_par, 0, 2);
			}
		}
	}
//		Hypercube_file.display_2_gaussiennes(modeles_parametres.grid_params, p, p, 0,2); //affiche cote à cote les données et le modèle

/*
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,0,1 , 2);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,0,2 , 3);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,1,0 , 4);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,1,1 , 5);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,1,2 , 6);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,2,0 , 7);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,2,1 , 8);
	Hypercube_file.display_avec_et_sans_regu(modeles_parametres.grid_params,2,2 , 9);
*/
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
				Hypercube_file.plot_line(modeles_parametres.grid_params, i, j, modeles_parametres.n_gauss);
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

		Hypercube_file.mean_parameters(modeles_parametres.grid_params, modeles_parametres.n_gauss);

		}




	}






