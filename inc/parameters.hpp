#ifndef DEF_PARAMETERS
#define DEF_PARAMETERS



#include "../L-BFGS-B-C/src/lbfgsb.h" //needed for the integer and logical type of the minimize_spec function
#include <iostream>
#include <stdio.h>
#include <cmath>
//#include <math.h>
#include <string>
#include <fstream>
#include <valarray>
#include <CCfits/CCfits>
#include <vector>


// mettre des const à la fin des déclarations si on ne modifie pas l'objet i.e. les attributs

/**
 * @brief Class associated to the values and choices given by the user in the parameters.txt file.
 *
 * We may use a DAT or a FITS file as the hypercube input file, we need to specify which file mode (DAT or FITS) we want to use. Some parameters are given as inputs for setulb(), the black box routine of L-BFGS-B-C used in the minimize() function.
 * Some of these variables such as the numbers n_gauss or the lambda_* are defined by the user and involved into the model as well.
 *
 *
 *
 */

template<typename T>
class parameters
{
	public:
	parameters();
	parameters(std::string str, std::string str2, std::string str3, std::string str4);
//	inline bool compare_ends(std::string const & value);
	inline bool is_true(std::string const & test);
	std::vector<int> dim_data; //inutile : file.dim_data
	int dim_x;
	int dim_y;
	int dim_v;
	
//	parameters
	int select_version; 
	std::string filename_dat; //!< Name of the DAT file to be used as hypercube input file.
	std::string filename_fits; //!< Name of the FITS file to be used as hypercube input file.
	std::string file_type_check;
	int slice_index_min; //!< Value of the minimum spectral index.
	int slice_index_max; //!< Value of the maximum spectral index.
	std::string fileout; //!< Name of the output file.
	std::string filename_noise; 
	int n_gauss; //!< Number of gaussians.
	T lambda_amp;
	T lambda_mu;
	T lambda_sig;
	T lambda_var_sig;
	T amp_fact_init;
	T sig_init;
	std::string init_option;	
	int maxiter_init;
	int maxiter;
	int m; //!< m is the maximum number of variable metric corrections to define the limited memory matrix.
	std::string check_noise;
	std::string check_regul;
	std::string check_descent;
	bool regul; //!< Activates regularization
	bool descent;
	int lstd;
	int ustd;
	long iprint; //!< Verbose mode for L-BFGS-B-C.
	long iprint_init;
	bool save_grid;
	T ub_sig;
	T lb_sig;
	T ub_sig_init;
	T lb_sig_init;
	bool jump_to_last_level;
	bool save_second_to_last_level;
	std::string second_to_last_level_grid_name;
	std::string double_or_float;
	bool float_mode, double_mode;
	std::string is_wrapper;
	bool wrapper;
	bool print_mean_parameters;
	bool noise;
	bool save_grid_through_multiresolution;
	std::vector<T> std_spect, mean_spect, max_spect, max_spect_norm;

	std::string name_without_extension;


	std::string input_format_fits_check;
	std::string output_format_fits_check;
	std::string noise_map_provided_check;
	std::string give_input_spectrum_check;
	std::string print_mean_parameters_check;
	bool input_format_fits;
	bool output_format_fits;
	bool noise_map_provided;
	bool give_input_spectrum;
	bool save_noise_map_in_fits;
	bool three_d_noise_mode;
};

#include "parameters.hpp"
#include <omp.h>
using namespace std;

template<typename T>
parameters<T>::parameters()
{
	//DUMMY CONSTRUCTOR
}

template<typename T>
parameters<T>::parameters(std::string str, std::string str2, std::string str3, std::string str4)
{
	std::string txt, egal;
    std::ifstream fichier(str, std::ios::in);  // on ouvre en lecture
    if(fichier){
		fichier >> txt >> egal >> this->input_format_fits_check;
		fichier >> txt >> egal >> this->filename_dat;
		fichier >> txt >> egal >> this->filename_fits;
		fichier >> txt >> egal >> this->output_format_fits_check;
		fichier >> txt >> egal >> this->fileout;
		fichier >> txt >> egal >> this->noise_map_provided_check;
		fichier >> txt >> egal >> this->filename_noise;
		fichier >> txt >> egal >> this->give_input_spectrum_check;
		fichier >> txt >> egal >> this->n_gauss;
		fichier >> txt >> egal >> this->lambda_amp;
		fichier >> txt >> egal >> this->lambda_mu;
		fichier >> txt >> egal >> this->lambda_sig;
		fichier >> txt >> egal >> this->lambda_var_sig;
		fichier >> txt >> egal >> this->amp_fact_init;
		fichier >> txt >> egal >> this->sig_init;
		fichier >> txt >> egal >> this->lstd;
		fichier >> txt >> egal >> this->ustd;
		fichier >> txt >> egal >> this->ub_sig;
		fichier >> txt >> egal >> this->lb_sig;
		fichier >> txt >> egal >> this->ub_sig_init;
		fichier >> txt >> egal >> this->lb_sig_init;
		fichier >> txt >> egal >> this->maxiter_init;
		fichier >> txt >> egal >> this->maxiter;
		fichier >> txt >> egal >> this->m;
		fichier >> txt >> egal >> this->init_option;
		fichier >> txt >> egal >> this->check_regul;
		fichier >> txt >> egal >> this->check_descent;
		fichier >> txt >> egal >> this->print_mean_parameters_check;
		fichier >> txt >> egal >> this->iprint;
		fichier >> txt >> egal >> this->iprint_init;

		if(is_true(input_format_fits_check)){
			this->input_format_fits = true;
		}else{
			this->input_format_fits = false;
		}

		if(is_true(output_format_fits_check)){
			this->output_format_fits = true;
		}else{
			this->output_format_fits = false;
		}

		if(is_true(noise_map_provided_check)){
			this->noise_map_provided = true;
		}else{
			this->noise_map_provided = false;
		}

		if(is_true(check_regul)){
			this->regul = true;
		}else{
			this->regul = false;
		}

		if(is_true(check_descent)){
			this->descent = true;
		}else{
			this->descent = false;
		}

		if(is_true(this->print_mean_parameters_check)){
			this->print_mean_parameters = true;
		}else{
			this->print_mean_parameters = false;
		}
		
		if(str2 == "-cpu" || str2 == "-CPU" || str2 == "-Cpu" || str2 == "-c" || str2 == "-C"){
			this->select_version = 0;
		}else if (str2 == "-gpu" || str2 == "-GPU" || str2 == "-Gpu" || str2 == "-G" || str2 == "-g"){
			this->select_version = 1;
		}else{
			this->select_version = 2;
		} 
//		printf("select_version = %d\n",this->select_version);
//		exit(0);
		if(str4 == "-double" || str4 == "-Double" || str4 == "-d" || str4 == "-D"){
			this->double_mode = true;
			this->float_mode = false;
		}else{
			this->double_mode = false;
			this->float_mode = true;
		}
		if(str3 == "-wrapper" || str3 == "-w" || str3 == "-Wrapper"){
			this->wrapper = true;
		}else{
			this->wrapper = false;
		}
	}else{
		std::cerr << "Can't open the parameters.txt file !" << std::endl;
	}
	fichier.close();

//	std::cout<<"fileout = "<<fileout<<std::endl;
	name_without_extension = fileout;
	if(this->output_format_fits){
	    name_without_extension.resize(name_without_extension.size() - 5);
	}else{
	    name_without_extension.resize(name_without_extension.size() - 4);
	}
//	std::cout<<"fileout = "<<fileout<<std::endl;

	this->save_grid_through_multiresolution = false; 
//	this->save_grid_through_multiresolution = true; 
	if(true){
		this->jump_to_last_level = true;
		this->save_second_to_last_level = false;
		this->second_to_last_level_grid_name = "right_before_last_level";
	}else{
		this->jump_to_last_level = false;
		this->save_second_to_last_level = true;
		this->second_to_last_level_grid_name = "right_before_last_level";
	}
}

template<typename T>
inline bool parameters<T>::is_true(std::string const & test)
{
    if (test == "true"||test == "True"||test == "TRUE"||test == "T"||test == "yes"||test == "Yes"||test == "YES"){
		return true;
	}else{
		return false;
	}
}

/*
template<typename T>
inline bool parameters<T>::compare_ends(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}
*/


#endif
