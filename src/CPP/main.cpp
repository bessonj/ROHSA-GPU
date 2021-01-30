#include "hypercube.hpp"
#include "parameters.hpp"
#include "algo_rohsa.hpp"
//#include "gradient.hpp"
#include <omp.h>
#include <cmath>
#include <iostream>

#include <string.h>


#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE1)*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE2)*(t##_SHAPE1)*x+(t##_SHAPE2)*y+z]

/** \mainpage Main Page
 *
 * \section comp Compiling and running the program
 *
 * We use CMake which is a cross-platform solftware managing the build process. First, create a build directory in ROHSA-GPU and go inside this directory.
 *
 * mkdir build && cd build
 *
 * Using CMake we can produce a Makefile. We may run it using make.
 *
 * cmake ../ && make
 *
 * When launching the program, we need to specify the parameters.txt file filled in by the user :  
 *
 * ./ROHSA-GPU parameters.txts
 *
 *
 * \section main main.cpp code
 *
 * \subsection par_txt Reading the user file parameters.txt
 *
 * We declare an parameters-type object that will the parameters.txt file. 
 *
 * parameters user_parametres(argv[1]);
 *
 *
 * \subsection hyp Getting the hypercube. 
 *
 * We can read the FITS or DAT file by using the class hypercube. 
 *
 * hypercube Hypercube_file(user_parametres, user_parametres.slice_index_min, user_parametres.slice_index_max, whole_data_in_cube); 
 *
 *
 * \subsection gauss_par Getting the gaussian parameters (call to the ROHSA algorithm). 
 *
 * The class algo_rohsa runs the ROHSA algorithm based on the two objects previously declared.
 *
 * algo_rohsa algo(user_parametres, Hypercube_file);
 *
 *
 * \subsection plot Plotting and storing the results.
 *
 * We can plot the smooth gaussian parameters maps and store the results back into a FITS file by using some of the routines of the class hypercube.
 *
 * algo_rohsa algo(user_parametres, Hypercube_file);
 *
 *
 *
 *
 *
 *
 *
 *
 */
/// Main function : it processes the FITS file, reads the parameters.txt, solves the optimization problem through the ROHSA algo and stores/print the result.
/// 
/// Details : 2 cases are distinguished : The data file is either a *.dat or a *.fits file.

template <typename T> 
void main_routine(parameters<T> &user_parametres){
	printf("Reading the cube ...\n");
    hypercube<T> Hypercube_file(user_parametres, user_parametres.slice_index_min, user_parametres.slice_index_max); 

	printf("Launching the ROHSA algorithm ...\n");
	algo_rohsa<T> algo(user_parametres, Hypercube_file);

	printf("Saving the result ...\n");
	Hypercube_file.save_result(algo.grid_params, user_parametres);
	printf("Result saved in dat file !\n");

	if(! user_parametres.noise_map_provided){
		printf("Saving noise_map ...\n");
		Hypercube_file.save_noise_map_in_fits(user_parametres, algo.std_data_map);
	}
	exit(0);
//	Hypercube_file.plot_line(algo.grid_params, **pos_x**, **pos_y**, user_parametres.n_gauss);
/*
	for(int num_gauss = 0; num_gauss < user_parametres.n_gauss; num_gauss ++){
		for (int num_par = 0; num_par < 3; num_par++)
		{
			Hypercube_file.display_avec_et_sans_regu(algo.grid_params, num_gauss, num_par , num_par+num_gauss*3);
		}
	}
*/
}

template void main_routine<double>(parameters<double>&);
template void main_routine<float>(parameters<float>&);


int main(int argc, char * argv[])
{
	parameters<float> user_parametres_float(argv[1], argv[2], argv[3], argv[4]);
	parameters<double> user_parametres_double(argv[1], argv[2], argv[3], argv[4]);

	if(user_parametres_double.double_mode){
		main_routine<double>(user_parametres_double);
	}else if(user_parametres_float.float_mode){
		main_routine<float>(user_parametres_float);
	}

/*

	if(user_parametres.file_type_fits){

		//Pour un FITS :
        hypercube<double> Hypercube_file(user_parametres, user_parametres.slice_index_min, user_parametres.slice_index_max, whole_data_in_cube, false); //true for reshaping, false for whole data
//        hypercube Hypercube_file(user_parametres, user_parametres.slice_index_min, user_parametres.slice_index_max, whole_data_in_cube); 

//		Hypercube_file.display_cube(0);
//
//		exit(0);
//Lecture des données, on regarde un extrait entre 2 indices
//set dimensions of the cube
//	Hypercube_file.display(Hypercube_file.data, 100); //affiche une tranche du tableau data à l'indice i=100
//	Hypercube_file.display_data(100); //affiche une tranche de data à l'indice 100
//	Hypercube_file.display_cube(100); //affiche une tranche de cube à l'indice 100

		double temps2_lecture = omp_get_wtime();

		algo_rohsa<double> algo(user_parametres, Hypercube_file);

        double temps2 = omp_get_wtime();

		Hypercube_file.save_result(algo.grid_params, user_parametres);

	std::cout<<"Temps de lecture : "<<temps2_lecture - temps1_lecture <<std::endl;
	std::cout<<"Temps total (hors enregistrement): "<<temps2 - temps1 <<std::endl;

		Hypercube_file.mean_parameters(algo.grid_params, user_parametres.n_gauss);


		//	for(int p = 40; p<70; p++){
		//		Hypercube_file.display_2_gaussiennes(user_parametres.grid_params, p, 0*3+1, 0, 1);
		//	}

			for(int num_gauss = 0; num_gauss < user_parametres.n_gauss; num_gauss ++){
				for (int num_par = 0; num_par < 3; num_par++)
				{
					Hypercube_file.display_avec_et_sans_regu(algo.grid_params, num_gauss, num_par , num_par+num_gauss*3);
				}
			}







exit(0);
//

	for(int p = 0; p<user_parametres.slice_index_max-user_parametres.slice_index_min-1; p++){

		for (int num_par = 0; num_par < 3; num_par++)
			{
			for(int num_gauss = 0; num_gauss < user_parametres.n_gauss; num_gauss ++)
				{
				for(int num_gauss_cur = 0; num_gauss_cur < user_parametres.n_gauss; num_gauss_cur ++)
					{
					if(num_gauss!=num_gauss_cur){
						Hypercube_file.display_2_gaussiennes_par_par_par(algo.grid_params, p, num_gauss*3+num_par, 3*num_gauss+num_par, 3*num_gauss_cur+num_par);
					}
				}
			}
		}
	}
		exit(0);
//for(int deuxieme_gauss=0; deuxieme_gauss<)
	for(int p = 0; p<Hypercube_file.cube[0][0].size(); p++){
		for(int num_gauss = 0; num_gauss < user_parametres.n_gauss; num_gauss ++){
			for (int num_par = 0; num_par < 3; num_par++)
			{
				Hypercube_file.display_2_gaussiennes(algo.grid_params, p, num_gauss*3+num_par, 0, 1);
			}
		}

		for(int num_gauss = 0; num_gauss < user_parametres.n_gauss; num_gauss ++){
			for (int num_par = 0; num_par < 3; num_par++)
			{
				Hypercube_file.display_2_gaussiennes(algo.grid_params,p , num_gauss*3+num_par, 1, 2);
			}
		}

			for(int num_gauss = 0; num_gauss < user_parametres.n_gauss; num_gauss ++){
			for (int num_par = 0; num_par < 3; num_par++)
			{
				Hypercube_file.display_2_gaussiennes(algo.grid_params, p, num_gauss*3+num_par, 0, 2);
			}
		}
	}
//		Hypercube_file.display_2_gaussiennes(user_parametres.grid_params, p, p, 0,2); //affiche cote à cote les données et le modèle


	}

	if(user_parametres.file_type_dat){

	//Pour un DAT :
    hypercube<double> Hypercube_file(user_parametres, user_parametres.slice_index_min, user_parametres.slice_index_max, whole_data_in_cube); 

//	        hypercube Hypercube_file(user_parametres); 
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

		algo_rohsa<double> algo(user_parametres, Hypercube_file);

        double temps2 = omp_get_wtime();
        std::cout<<"Temps de lecture : "<<temps2_lecture - temps1_lecture <<std::endl;
        std::cout<<"Temps total (hors enregistrement): "<<temps2 - temps1 <<std::endl;


		int i_max = 4;
		int j_max = 5;

		Hypercube_file.save_result(algo.grid_params, user_parametres);

	for(int num_gauss = 0; num_gauss < user_parametres.n_gauss; num_gauss ++){
		for (int num_par = 0; num_par < 3; num_par++)
		{
			Hypercube_file.display_avec_et_sans_regu(algo.grid_params, num_gauss, num_par , num_par+num_gauss*3);
		}
	}



	exit(0);
		for(int i=0;i<i_max;i++){
			for(int j=0;j<j_max;j++){
				Hypercube_file.plot_line(algo.grid_params, i, j, user_parametres.n_gauss);
			}
		}

//	Hypercube_file.display_result(user_parametres.grid_params, 100, user_parametres.n_gauss); //affiche une tranche du cube reconstitué du modèle à l'indice 100
//	Hypercube_file.display_cube(100); //affiche une tranche de cube à l'indice 100

		for(int p = 0; p<user_parametres.slice_index_max-user_parametres.slice_index_min-1; p++){
			Hypercube_file.display_result_and_data(algo.grid_params, p, user_parametres.n_gauss, true); //affiche cote à cote les données et le modèle
		}


	Hypercube_file.display_avec_et_sans_regu(algo.grid_params,0,0 , 1);
	Hypercube_file.display_avec_et_sans_regu(algo.grid_params,0,1 , 2);
	Hypercube_file.display_avec_et_sans_regu(algo.grid_params,0,2 , 3);
	Hypercube_file.display_avec_et_sans_regu(algo.grid_params,1,0 , 4);
	Hypercube_file.display_avec_et_sans_regu(algo.grid_params,1,1 , 5);
	Hypercube_file.display_avec_et_sans_regu(algo.grid_params,1,2 , 6);
	Hypercube_file.display_avec_et_sans_regu(algo.grid_params,2,0 , 7);
	Hypercube_file.display_avec_et_sans_regu(algo.grid_params,2,1 , 8);
	Hypercube_file.display_avec_et_sans_regu(algo.grid_params,2,2 , 9);

		Hypercube_file.mean_parameters(algo.grid_params, user_parametres.n_gauss);

		}
*/
	}
