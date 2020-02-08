#ifndef DEF_MODEL
#define DEF_MODEL

#include "./L-BFGS-B-C/src/lbfgsb.h" //needed for the integer and logical type of the minimize_spec function
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

class model
{
	public:

	model();
	void write_model_in_binary();


	std::vector<std::vector<double>> kernel;
	std::vector<int> dim_data; //inutile : file.dim_data
	int dim_x;
	int dim_y;
	int dim_v;

	int n_gauss_add; //EN DISCUTER AVEC ANTOINE
	

//	parameters
	std::string filename_dat;
	std::string filename_fits;
	std::string file_type_dat_check;
	std::string file_type_fits_check;
	int slice_index_min;
	int slice_index_max;
	bool file_type_dat;
	bool file_type_fits;
	std::string fileout;
	std::string filename_noise;
	int n_gauss;
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
	int m;
	std::string check_noise;
	std::string check_regul;
	std::string check_descent;
	bool noise;
	bool regul;
	bool descent;
	int lstd;
	int ustd;
	long iprint;
	int iprint_init;
	std::string check_save_grid;
	bool save_grid;
	double ub_sig;
	double lb_sig;
	double ub_sig_init;
	double lb_sig_init;

	std::vector<double> std_spect, mean_spect, max_spect, max_spect_norm;
	std::vector<std::vector<std::vector<double>>> grid_params; // sortie, paramètres du modèle
	std::vector<std::vector<std::vector<double>>> fit_params; // paramètres du modèle

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
