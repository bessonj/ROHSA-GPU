#include "hypercube.hpp"
#include "parameters.hpp"
#include "algo_rohsa.hpp"
//#include "gradient.hpp"
#include <omp.h>
#include <cmath>
#include <iostream>
#include <string.h>
#include <stdio.h>

#include "fortran_to_cpp_conversion.h"

////VARIABLES TO BE DEFINED BY USER IF REQUIRED////
////LEAVE BLANK IF YOU ARE ALREADY USING A PRE-PROCESSED DAT FILE////

//PRE-PROCESSING OF THE FITS IMAGE
//(defines pre-processing parameters for the fits image)

//--> Choose between 2 modes :
// Either extracting a square centered on a given pixel 
// whose size is specified below or using the whole cube
// First mode :
#define EXTRACT_SQUARE_CENTERED_ON_PIX false
//~>if EXTRACT_SQUARE_CENTERED_ON_PIX is false, 
//  leave other lines below with dummy values
	#define SQUARE_SIZE 1024
	#define PIXEL_AUTOMATICALLY_CENTERED false //
	#define PIX_X 10
	#define PIX_Y 10
	//Spectral range :
	#define MIN_INDEX_RANGE_CHANNEL 64
	#define MAX_INDEX_RANGE_CHANNEL 123
	//~>end

// Second mode :
#define USE_WHOLE_FITS_FILE true


////OTHER OPTIONS 

//If true : stores the std_map in a fits file.
//(provided it's computed on the fly)
#define SAVE_STD_MAP_IN_FITS false 

//Activate 3D noise mode
#define THREE_D_NOISE_MODE false



////MACRO
#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE1)*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE2)*(t##_SHAPE1)*x+(t##_SHAPE2)*y+z]

double temps_test = 0;
//extern void test1(int n);

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


//	hypercube<T> Hypercube_file(user_parameters);
/*
	hypercube<T> Hypercube_file(1,1,1);
	std::vector<T> newVector = Hypercube_file.read_vector_from_file("right_before_last_level.raw");
	printf("newVector.size() = %d\n", newVector.size());
	int dim_0 = 36;
	int dim_1 = 1024;
	int dim_2 = 1024;
	std::vector <std::vector<std::vector<T>>> grid(dim_0, std::vector<std::vector<T>>(dim_1, std::vector<T>(dim_2,0.)));
	for(int k=0; k<dim_0; k++)
	{
		for(int j=0; j<dim_1; j++)
		{
			for(int i=0; i<dim_2; i++)
			{
				grid[k][j][i] = newVector[k*dim_2*dim_1+j*dim_2+i];
			}
		}
	}

	Hypercube_file.save_grid_in_fits(user_parameters, grid);
	exit(0);
*/
/*
	for(int i = 0; i<newVector.size(); i++){
		printf("newVector[%d] = %f\n", i, newVector[i]);
	}
*/

template <typename T> 
void main_routine(parameters<T> &user_parameters){

	/*
	////RECOVERS THE FITS FILE FROM THE RAW ONE
	hypercube<T> Hypercube_file(1,1,1);
	std::vector<T> newVector = Hypercube_file.read_vector_from_file("DHIGLS_UM_Tb_gauss_run_c_1024_hybrid_lambda_gauss_8_lambda_0dot1.raw");
	printf("newVector.size() = %d\n", newVector.size());
	int dim_0 = 24;
	int dim_1 = 1024;
	int dim_2 = 1024;
	std::vector <std::vector<std::vector<T>>> grid(dim_0, std::vector<std::vector<T>>(dim_1, std::vector<T>(dim_2,0.)));
	for(int k=0; k<dim_0; k++)
	{
		for(int j=0; j<dim_1; j++)
		{
			for(int i=0; i<dim_2; i++)
			{
				grid[k][j][i] = newVector[k*dim_2*dim_1+j*dim_2+i];
			}
		}
	}
	printf("before saving !\n");

	Hypercube_file.save_grid_in_fits(user_parameters, grid);
	exit(0);
	*/
/*
	hypercube<T> Hypercube_file(user_parameters, user_parameters.slice_index_min, user_parameters.slice_index_max, 730, 650, SQUARE_SIZE, false, true, false);

	Hypercube_file.save_grid_in_fits(user_parameters, algo.grid_params);

	exit(0);
    hypercube<T> Hypercube_file(user_parameters, user_parameters.slice_index_min, user_parameters.slice_index_max); 
	std::vector<std::vector<T>> std_map_init(Hypercube_file.dim_data[1], std::vector<T>(Hypercube_file.dim_data[0],0.));
	algo_rohsa<T> algo(std_map_init, user_parameters, Hypercube_file);
//	algo.set_stdmap_transpose(std_map_init, Hypercube_file.data, user_parameters.lstd, user_parameters.ustd);
	Hypercube_file.save_noise_map_in_fits(user_parameters, std_map_init);

	exit(0);
*/
	user_parameters.slice_index_min = MIN_INDEX_RANGE_CHANNEL;
	user_parameters.slice_index_max = MAX_INDEX_RANGE_CHANNEL;
	user_parameters.save_noise_map_in_fits = SAVE_STD_MAP_IN_FITS;
	user_parameters.three_d_noise_mode = THREE_D_NOISE_MODE;
//	user_parameters.size_side_square = SQUARE_SIZE;

	printf("Reading the cube ...\n");
	if (user_parameters.input_format_fits){
		hypercube<T> Hypercube_file(user_parameters, user_parameters.slice_index_min, user_parameters.slice_index_max, 730, 650, SQUARE_SIZE, false, true, false);
		printf("Launching the ROHSA algorithm ...\n");
		algo_rohsa<T> algo(user_parameters, Hypercube_file);

		const std::string filename_raw = user_parameters.name_without_extension+".raw";
		std::cout<<"filename_raw = "<<filename_raw<<std::endl;	

		printf("Saving the result ...\n");
		Hypercube_file.save_grid_in_fits(user_parameters, algo.grid_params);
/*		try{
			throw 
		}
*/
//		Hypercube_file.write_in_file(algo.grid_params, filename_raw);
	//	Hypercube_file.save_result(algo.grid_params, user_parameters);

	//	if(user_parameters.output_format_fits){
	//		Hypercube_file.save_grid_in_fits(user_parameters, algo.grid_params);
	//		printf("Result saved in fits file !\n");
	//	}else{
	//		Hypercube_file.save_result(algo.grid_params, user_parameters);
	//		printf("Result saved in dat file !\n");
	//	}

	if(user_parameters.save_noise_map_in_fits){
/*
			printf("Saving noise_map ...\n");
			Hypercube_file.save_noise_map_in_fits(user_parameters, algo.std_data);
*/
//			printf("Noise_map saved in fits file!\n");
		}
	}else{
	    hypercube<T> Hypercube_file(user_parameters, user_parameters.slice_index_min, user_parameters.slice_index_max); 
		printf("Launching the ROHSA algorithm ...\n");
		algo_rohsa<T> algo(user_parameters, Hypercube_file);

		const std::string filename_raw = user_parameters.name_without_extension+".raw";
		std::cout<<"filename_raw = "<<filename_raw<<std::endl;	

		printf("Saving the result ...\n");
//		Hypercube_file.write_in_file(algo.grid_params, filename_raw);
	//	Hypercube_file.save_result(algo.grid_params, user_parameters);
		Hypercube_file.save_grid_in_fits(user_parameters, algo.grid_params);

	//	if(user_parameters.output_format_fits){
	//		Hypercube_file.save_grid_in_fits(user_parameters, algo.grid_params);
	//		printf("Result saved in fits file !\n");
	//	}else{
	//		Hypercube_file.save_result(algo.grid_params, user_parameters);
	//		printf("Result saved in dat file !\n");
	//	}


		if(user_parameters.save_noise_map_in_fits){
			printf("Saving noise_map ...\n");
			Hypercube_file.save_noise_map_in_fits(user_parameters, algo.std_data);
			printf("Noise_map saved in fits file!\n");
		}
	}
/*
	printf("Launching the ROHSA algorithm ...\n");
	algo_rohsa<T> algo(user_parameters, Hypercube_file);

	const std::string filename_raw = user_parameters.name_without_extension+".raw";
	std::cout<<"filename_raw = "<<filename_raw<<std::endl;	

	printf("Saving the result ...\n");
	Hypercube_file.write_in_file(algo.grid_params, filename_raw);
//	Hypercube_file.save_result(algo.grid_params, user_parameters);
	Hypercube_file.save_grid_in_fits(user_parameters, algo.grid_params);

//	if(user_parameters.output_format_fits){
//		Hypercube_file.save_grid_in_fits(user_parameters, algo.grid_params);
//		printf("Result saved in fits file !\n");
//	}else{
//		Hypercube_file.save_result(algo.grid_params, user_parameters);
//		printf("Result saved in dat file !\n");
//	}


	if(user_parameters.save_noise_map_in_fits){
		printf("Saving noise_map ...\n");
		Hypercube_file.save_noise_map_in_fits(user_parameters, algo.std_data);
		printf("Noise_map saved in fits file!\n");
	}
	printf("temps_test = %f\n");
*/
}

template void main_routine<double>(parameters<double>&);
//template void main_routine<float>(parameters<float>&);

void mygrad_spec_double(int dim_v,double gradient[], std::vector<double> &residual, double params[], int n_gauss_i) {
	std::vector<std::vector<double>> dF_over_dB(3*n_gauss_i, std::vector<double>(dim_v,0.));
//	double g = 0.;

	int i,k;
/*
	for(int p(0); p<dim_v; p++) {
		printf("residual[%d] = %.27f\n",p,residual[p]);
	}
*/
	for(int p(0); p<3*n_gauss_i; p++) {
		gradient[p]=0.;
	}

	for(i=0; i<n_gauss_i; i++) {
		for(int k(0); k<dim_v; k++) {
			double nu = double(k+1);
			dF_over_dB[0+3*i][k] += exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

			dF_over_dB[1+3*i][k] +=  params[3*i]*( nu - params[1+3*i])/pow(params[2+3*i],2.) * exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

			dF_over_dB[2+3*i][k] += params[3*i]*pow( nu - params[1+3*i] , 2.)/(pow(params[2+3*i],3.)) * exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

		}
	}
	for(i=0; i<n_gauss_i; i++) {
		for(int k(0); k<dim_v; k++) {
//			double nu = double(k+1);
//			printf("nombre = %.27f\n", exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) ));
			gradient[0+3*i] += dF_over_dB[0+3*i][k]*residual[k];

			gradient[1+3*i] += dF_over_dB[1+3*i][k]*residual[k];

			gradient[2+3*i] += dF_over_dB[2+3*i][k]*residual[k];

		}
	}	
	/*
	for(i=0; i<n_gauss_i; i++) {
		for(int k(0); k<dim_v; k++) {
			double nu = double(k+1);
//			printf("nombre = %.27f\n", exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) ));
			gradient[0+3*i] += exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) )*residual[k];

			gradient[1+3*i] +=  params[3*i]*( nu - params[1+3*i])/pow(params[2+3*i],2.) * exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) )*residual[k];

			gradient[2+3*i] += params[3*i]*pow( nu - params[1+3*i] , 2.)/(pow(params[2+3*i],3.)) * exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) )*residual[k];

		}
	}
	*/
}

void test(char * argv[]){
/*
slice_index_min = 0
slice_index_max = 199
*/
/*
	std::cout<<"setprecision() = "<<std::setprecision(27)<<std::endl;
	std::vector<double> line(10,0);
	std::vector<double> residual(10,0);

	double g[9];
	double params[9];
	for(int i(0); i<9; i++) {
		g[i]=0.;
    }

params[        0] =  1.000000000000000000000000000;
params[        1] =  10.000000000000000000000000000;
params[        2] =  10.000000000000000000000000000;
params[        3] =  1.000000000000000000000000000;
params[        4] =  10.000000000000000000000000000;
params[        5] =  10.000000000000000000000000000;
params[        6] =  1.000000000000000000000000000;
params[        7] =  10.000000000000000000000000000;
params[        8] =  10.000000000000000000000000000;

residual[        0] = 1.000000000000000000000000000;
residual[        1] = 1.000000000000000000000000000;
residual[        2] = 1.000000000000000000000000000;
residual[        3] = 1.000000000000000000000000000;
residual[        4] = 1.000000000000000000000000000;
residual[        5] = 1.000000000000000000000000000;
residual[        6] = 1.000000000000000000000000000;
residual[        7] = 1.000000000000000000000000000;
residual[        8] = 1.000000000000000000000000000;
residual[        9] = 1.000000000000000000000000000;


	mygrad_spec_double(10, g, residual, params, 3);

	for(int p = 0; p<10; p++){
		printf("residual[%d] = %.27f\n", p, residual[p]);
	}

	for(int p = 0; p<9; p++){
		printf("params[%d] = %.27f\n", p, params[p]);
	}

	for(int p = 0; p<9; p++){
		printf("g[%d] = %.27f\n", p, g[p]);
	}


	std::cout<<"------------------------------------"<<std::endl;




//	free(g);

	exit(0);
*/


	std::cout<<"setprecision() = "<<std::setprecision(27)<<std::endl;
//	test();
    printf("Hello from C!\n");

	parameters<double> M_user_parametres_double(argv[1], argv[2], argv[3], argv[4]);
	int dim_x = 4;
	int dim_y = 4;
	int dim_v = 10;
	int n_gauss = 3;
	M_user_parametres_double.n_gauss = n_gauss;
	M_user_parametres_double.lambda_amp = 1.0;
	M_user_parametres_double.lambda_mu = 1.0;
	M_user_parametres_double.lambda_sig = 1.0;
	M_user_parametres_double.lambda_var_sig = 1.0;
	int n = dim_x*dim_y*n_gauss*3;

	double f = 0.;

	std::vector<std::vector<std::vector<double>>> cube(dim_x, std::vector<std::vector<double>>(dim_y, std::vector<double>(dim_v,1.0)));
	std::vector<std::vector<double>> std_map_vector(dim_x, std::vector<double>(dim_y, 1.0));

    double* temps = NULL;
    temps = (double*)malloc(4*sizeof(double));
    temps[0] = 0.;
    temps[1] = 0.;
    temps[2] = 0.;
    temps[3] = 0.;
    temps[4] = 0.;

	int n_beta = (3*n_gauss*dim_x*dim_y+n_gauss);

	double* beta = NULL;
	beta = (double*)malloc(n_beta*sizeof(double));
	for(int i(0); i<n_beta; i++) {
		beta[i]=1.0;
    }

	double* g = NULL;
	g = (double*)malloc(n_beta*sizeof(double));
	for(int i(0); i<n_beta; i++) {
		g[i]=0.;
    }

    double* std_map = NULL;
    std_map = (double*)malloc(dim_x*dim_y*sizeof(double));
	for(int i=0; i<dim_y; i++){
		for(int j=0; j<dim_x; j++){
			std_map[i*dim_x+j]=1.0;
//			std_map_[j*dim_x+i]=std_map[i][j];
		}
	}
	double* cube_flattened = NULL;
	size_t size_cube = dim_x*dim_y*dim_v*sizeof(double);
	cube_flattened = (double*)malloc(size_cube);
	for(int i=0; i<dim_x*dim_y*dim_v; i++) {
		cube_flattened[i] = 1.0;
	}

	double temps_transfert_d = 0.;
	double temps_mirroirs = 0.;
	double* temps_detail_regu;
	
//template void f_g_cube_fast_unidimensional(parameters<double>&, double&, double*, int, double*, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, double*, double*);	

//--> cpu 
/*
	f_g_cube_fast_unidimensional<double>(M_user_parametres_double, f, g, n_beta, cube_flattened, cube, beta, dim_v, dim_y, dim_x, std_map, temps);
	for(int p = 0; p<n_beta; p++){
		printf("g[%d] = %.27f\n", p, g[p]);
	}
*/

//--> gpu code for f_g_cube_cuda_L_clean :

	f_g_cube_cuda_L_clean<double>(M_user_parametres_double, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map_vector, cube_flattened, temps, temps_transfert_d, temps_mirroirs, temps_detail_regu); // expérimentation gradient
			for(int i = 0; i<dim_x; i++){
		for(int j = 0; j<dim_y; j++){
	for(int p = 0; p<3*n_gauss; p++){
	
		printf("g[%d] = %.27f\n", p*dim_x*dim_y+j*dim_x+i, g[p*dim_x*dim_y+j*dim_x+i]);

			}
		}
	}

exit(0);

//	free(g);
	free(temps);
	free(std_map);
	free(cube_flattened);
	free(beta);
	free(g);

	exit(0);


/*
   int input = 3;
   int output = 0;
   printf("TEST 1 !\n");
   square_(&input,&output); 
   printf("TEST 2 !\n");
   std::cout<< output << std::endl;  // returns 9 
	double f = 3.14159265;
	print_a_real_(&f);

	int n = 3;
	double* tab = NULL;
	tab = (double*)malloc(n*sizeof(double));
	tab[0] = 1.1;
	tab[1] = 1.2;
	tab[2] = 1.3;

	print_an_array_(&n, tab);

	char char_test[] = "blablabla";
	std::cout<<"Read char_test c++ -> "<<char_test<<std::endl;
//	unsigned char *char_test;// = 'blablabla';

	print_a_char_(char_test);
//    fortransub();

    int logical_test[] = {1, 0, 1, 0};
	std::cout<<"logical_test[0] = "<<logical_test[0]<<std::endl;
	std::cout<<"logical_test[1] = "<<logical_test[1]<<std::endl;
	std::cout<<"logical_test[2] = "<<logical_test[2]<<std::endl;
	std::cout<<"logical_test[3] = "<<logical_test[3]<<std::endl;
	print_a_logical_(logical_test);
*/

//	minimize();
//   	exit(0);

}

int main(int argc, char * argv[])
{
//	test(argv);

	std::cout<<"argv = "<<argv[1]<<" , "<<argv[2]<<" , "<<argv[3]<<" , "<<argv[4]<<std::endl;
	parameters<float> user_parametres_float(argv[1], argv[2], argv[3], argv[4]);
	parameters<double> user_parametres_double(argv[1], argv[2], argv[3], argv[4]);

	if(user_parametres_double.double_mode){
		main_routine<double>(user_parametres_double);
	}else if(user_parametres_float.float_mode){
//		main_routine<float>(user_parametres_float);
	}
}

