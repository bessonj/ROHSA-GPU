#include "hypercube.hpp"
#include "parameters.hpp"
#include "algo_rohsa.hpp"
//#include "gradient.hpp"
#include <omp.h>
#include <cmath>
#include <iostream>
#include <string.h>

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
    hypercube<T> Hypercube_file(user_parameters, user_parameters.slice_index_min, user_parameters.slice_index_max); 
//	hypercube<T> Hypercube_file(user_parameters, user_parameters.slice_index_min, user_parameters.slice_index_max, 730, 650, SQUARE_SIZE, false, true, false);
//	cudaSetDevice(0);
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
}

template void main_routine<double>(parameters<double>&);
//template void main_routine<float>(parameters<float>&);


int main(int argc, char * argv[])
{
/*
slice_index_min = 0
slice_index_max = 199
*/

	std::cout<<"argv = "<<argv[1]<<" , "<<argv[2]<<" , "<<argv[3]<<" , "<<argv[4]<<std::endl;
	parameters<float> user_parametres_float(argv[1], argv[2], argv[3], argv[4]);
	parameters<double> user_parametres_double(argv[1], argv[2], argv[3], argv[4]);

	if(user_parametres_double.double_mode){
		main_routine<double>(user_parametres_double);
	}else if(user_parametres_float.float_mode){
//		main_routine<float>(user_parametres_float);
	}
}
