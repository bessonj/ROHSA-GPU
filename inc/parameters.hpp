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
		inline bool compare_ends(std::string const & value, std::string const & ending);
//		copy_double_T(parameters<double> &M_d, parameters<T> &M);
		std::vector<int> dim_data; //inutile : file.dim_data
		int dim_x;
		int dim_y;
		int dim_v;
		int n_gauss_add; 
	
//	parameters
	int select_version; 
	std::string filename_dat; //!< Name of the DAT file to be used as hypercube input file.
	std::string filename_fits; //!< Name of the FITS file to be used as hypercube input file.
	std::string file_type_dat_check;
	std::string file_type_fits_check;
	int slice_index_min; //!< Value of the minimum spectral index.
	int slice_index_max; //!< Value of the maximum spectral index.
	bool file_type_dat;
	bool file_type_fits;
	std::string fileout; //!< Name of the output file.
	std::string filename_noise; 
	int n_gauss; //!< Number of gaussians.
	T lambda_amp;
	T lambda_mu;
	T lambda_sig;
	T lambda_var_amp;
	T lambda_var_mu;
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
	std::string check_save_grid;
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
	bool noise_map_provided;
	bool noise;

	std::vector<T> std_spect, mean_spect, max_spect, max_spect_norm;

};

#include "parameters.hpp"
#include <omp.h>
using namespace std;

template<typename T>
parameters<T>::parameters()
{
/*
	n_gauss_add = 0;

	std::string txt, egal;
        std::ifstream fichier("parameters.txt", std::ios::in);  // on ouvre en lecture
 
        if(fichier)  // si l'ouverture a fonctionné
        {
		fichier >> txt >> egal >> filename_dat;
		fichier >> txt >> egal >> filename_fits;
		fichier >> txt >> egal >> file_type_dat_check;
		fichier >> txt >> egal >> file_type_fits_check;
		fichier >> txt >> egal >> slice_index_min;
		fichier >> txt >> egal >> slice_index_max;
		fichier >> txt >> egal >> fileout;
		fichier >> txt >> egal >> filename_noise;
		fichier >> txt >> egal >> n_gauss;
		fichier >> txt >> egal >> lambda_amp;
		fichier >> txt >> egal >> lambda_mu;
		fichier >> txt >> egal >> lambda_sig;
		fichier >> txt >> egal >> lambda_var_amp;
		fichier >> txt >> egal >> lambda_var_mu;
		fichier >> txt >> egal >> lambda_var_sig;
		fichier >> txt >> egal >> amp_fact_init;
		fichier >> txt >> egal >> sig_init;
		fichier >> txt >> egal >> init_option;
		fichier >> txt >> egal >> maxiter_init;
		fichier >> txt >> egal >> maxiter;
		fichier >> txt >> egal >> m;
		fichier >> txt >> egal >> check_noise;
		fichier >> txt >> egal >> check_regul;
		fichier >> txt >> egal >> check_descent;
		fichier >> txt >> egal >> lstd;
		fichier >> txt >> egal >> ustd;
		fichier >> txt >> egal >> iprint;
		fichier >> txt >> egal >> iprint_init;
		fichier >> txt >> egal >> check_save_grid;
		fichier >> txt >> egal >> ub_sig;
		fichier >> txt >> egal >> lb_sig;
		fichier >> txt >> egal >> ub_sig_init;
		fichier >> txt >> egal >> lb_sig_init;
		fichier >> txt >> egal >> double_or_float;
		fichier >> txt >> egal >> is_wrapper;
		if(file_type_dat_check == "true")
			file_type_dat = true;
		else 
			file_type_dat = false;
		if(file_type_fits_check == "true")
			file_type_fits = true;
		else 
			file_type_fits = false;
		if(check_save_grid == "true")
			save_grid = true;
		else 
			save_grid = false;		
		if(check_noise == "true")
			noise = true;
		else
			noise = false;
		if(check_regul == "true")
			regul = true;
		else
			regul = false;
		if(check_descent == "true")
			descent = true;
		else
			descent = false;
		if(double_or_float == "double" || double_or_float == "Double" || double_or_float == "d" || double_or_float == "D"){
			double_mode = true;
			float_mode = false;
		}else{
			double_mode = false;
			float_mode = true;
		}if(is_wrapper == "wrapper" || is_wrapper == "w" || is_wrapper == "Wrapper"){
			wrapper = true;
		}else{
			wrapper = false;
		}
        }else	
            std::cerr << "Can't open the parameters.txt file !" << std::endl;
		fichier.close();
*/
}

template<typename T>
parameters<T>::parameters(std::string str, std::string str2, std::string str3, std::string str4)
{
	n_gauss_add = 0;

	std::string txt, egal;
        std::ifstream fichier(str, std::ios::in);  // on ouvre en lecture
 
        if(fichier)  // si l'ouverture a fonctionné
        {
		fichier >> txt >> egal >> filename_dat;
		fichier >> txt >> egal >> filename_fits;
		fichier >> txt >> egal >> file_type_dat_check;
		fichier >> txt >> egal >> file_type_fits_check;
		fichier >> txt >> egal >> slice_index_min;
		fichier >> txt >> egal >> slice_index_max;
		fichier >> txt >> egal >> fileout;
		fichier >> txt >> egal >> filename_noise;
		fichier >> txt >> egal >> n_gauss;
		fichier >> txt >> egal >> lambda_amp;
		fichier >> txt >> egal >> lambda_mu;
		fichier >> txt >> egal >> lambda_sig;
		fichier >> txt >> egal >> lambda_var_amp;
		fichier >> txt >> egal >> lambda_var_mu;
		fichier >> txt >> egal >> lambda_var_sig;
		fichier >> txt >> egal >> amp_fact_init;
		fichier >> txt >> egal >> sig_init;
		fichier >> txt >> egal >> init_option;
		fichier >> txt >> egal >> maxiter_init;
		fichier >> txt >> egal >> maxiter;
		fichier >> txt >> egal >> m;
		fichier >> txt >> egal >> check_noise;
		fichier >> txt >> egal >> check_regul;
		fichier >> txt >> egal >> check_descent;
		fichier >> txt >> egal >> lstd;
		fichier >> txt >> egal >> ustd;
		fichier >> txt >> egal >> iprint;
		fichier >> txt >> egal >> iprint_init;
		fichier >> txt >> egal >> check_save_grid;
		fichier >> txt >> egal >> ub_sig;
		fichier >> txt >> egal >> lb_sig;
		fichier >> txt >> egal >> ub_sig_init;
		fichier >> txt >> egal >> lb_sig_init;
		fichier >> txt >> egal >> print_mean_parameters;
		
		if(file_type_dat_check == "true")
			file_type_dat = true;
		else 
			file_type_dat = false;
		if(file_type_fits_check == "true")
			file_type_fits = true;
		else 
			file_type_fits = false;
		if(check_save_grid == "true")
			save_grid = true;
		else 
			save_grid = false;		
		if(check_noise == "true")
			noise = true;
		else
			noise = false;
		if(check_regul == "true")
			regul = true;
		else
			regul = false;
		if(check_descent == "true")
			descent = true;
		else
			descent = false;
		if(str2 == "-cpu" || str2 == "-CPU" || str2 == "-Cpu" || str2 == "-c" || str2 == "-C")
			select_version = 0;
		else if (str2 == "-gpu" || str2 == "-GPU" || str2 == "-Gpu" || str2 == "-G" || str2 == "-g")
			select_version = 1;
		else 
			select_version = 2;
		if(str4 == "-double" || str4 == "-Double" || str4 == "-d" || str4 == "-D"){
			double_mode = true;
			float_mode = false;
		}else{
			double_mode = false;
			float_mode = true;
		}if(str3 == "-wrapper" || str3 == "-w" || str3 == "-Wrapper"){
			wrapper = true;
		}else{
			wrapper = false;
		}
        }else
            std::cerr << "Can't open the parameters.txt file !" << std::endl;
		fichier.close();

	std::string comp (".fits");
	std::string COMP (".FITS");
	bool is_fits_caps_lock = compare_ends(this->filename_noise, COMP);
	bool is_fits = compare_ends(this->filename_noise, comp);
	if(is_fits || is_fits_caps_lock){
		this->noise_map_provided = true;
	} else {
		this->noise_map_provided = false;
	}
	//overwrite verification
	this->noise_map_provided = this->noise;


	if(false){
//	if(true){
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
inline bool parameters<T>::compare_ends(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}



#endif
