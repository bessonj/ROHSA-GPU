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

class parameters
{
	public:

	parameters();
	parameters(std::string str);
	void write_in_binary();

	std::vector<std::vector<double>> kernel;
	std::vector<int> dim_data; //inutile : file.dim_data
	int dim_x;
	int dim_y;
	int dim_v;

	int n_gauss_add; 
	

//	parameters
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
	double lambda_amp;
	double lambda_mu;
	double lambda_sig;
	double lambda_var_amp;
	double lambda_var_mu;
	double lambda_var_sig;
	double amp_fact_init;
	double sig_init;
	std::string init_option;	
	int maxiter_init;
	int maxiter;
	int m; //!< m is the maximum number of variable metric corrections to define the limited memory matrix.
	std::string check_noise;
	std::string check_regul;
	std::string check_descent;
	bool noise;
	bool regul; //!< Activates regularization
	bool descent;
	int lstd;
	int ustd;
	long iprint; //!< Verbose mode for L-BFGS-B-C.
	int iprint_init;
	std::string check_save_grid;
	bool save_grid;
	double ub_sig;
	double lb_sig;
	double ub_sig_init;
	double lb_sig_init;

	std::vector<double> std_spect, mean_spect, max_spect, max_spect_norm;

/*
	int n_gauss_add;
	int nside;
	int n;
	int power;
	double ub_sig_init;
	double ub_sig;

	std::vector<int> dim_data;
	std::vector<int> dim_cube; 

*/
};

#endif
