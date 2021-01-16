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


int main(int argc, char * argv[])
{
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

	double temps1 = omp_get_wtime();
	double temps1_lecture = omp_get_wtime();

//	parameters<double> user_parametres(argv[1], argv[2]);


	parameters<float> user_parametres_float(argv[1], argv[2], argv[3], argv[4]);
//    hypercube<double> Hypercube_file_float(user_parametres_float, user_parametres_float.slice_index_min, user_parametres_float.slice_index_max, 540, 375, 256); 
    hypercube<float> Hypercube_file_float(user_parametres_float, user_parametres_float.slice_index_min, user_parametres_float.slice_index_max); 

//	Hypercube_file_float.get_noise_map_from_fits(user_parametres_float);

//    hypercube<float> Hypercube_file_float(user_parametres_float, user_parametres_float.slice_index_min, user_parametres_float.slice_index_max); 
/*
	for(int ind = 0; ind<100; ind++){
		Hypercube_file_float.display_data(ind);
	}
*/
	algo_rohsa<float> algo_float(user_parametres_float, Hypercube_file_float);
	Hypercube_file_float.save_result(algo_float.grid_params, user_parametres_float);

	printf("result saved !\n");
	Hypercube_file_float.plot_line(algo_float.grid_params,127,127,12);
	Hypercube_file_float.plot_line(algo_float.grid_params,128,127,12);
	Hypercube_file_float.plot_line(algo_float.grid_params,127,128,12);
	Hypercube_file_float.plot_line(algo_float.grid_params,128,128,12);
	Hypercube_file_float.plot_line(algo_float.grid_params,129,128,12);
	Hypercube_file_float.plot_line(algo_float.grid_params,129,129,12);
	Hypercube_file_float.plot_line(algo_float.grid_params,127,128,12);
	Hypercube_file_float.plot_line(algo_float.grid_params,128,129,12);


	for(int num_gauss = 0; num_gauss < user_parametres_float.n_gauss; num_gauss ++){
		for (int num_par = 0; num_par < 3; num_par++)
		{
			Hypercube_file_float.display_avec_et_sans_regu(algo_float.grid_params, num_gauss, num_par , num_par+num_gauss*3);
		}
	}

	exit(0);
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






