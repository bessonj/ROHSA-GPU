#include "Parse.h"
#include "Tool.h"


int main()
{
  /*
	bool regul;
	bool descent;
	bool save_grid;
	int n_gauss_add;
	int n_gauss;
	int nside;
	int n;
	int m;
	int power;
	int lstd;
	int ustd;
	int iprint;
	int iprint_init;
	int maxiter;
	int maxiter_init;
	double lambda_amp;
	double  lambda_mu;
	double lambda_sig;
	double lambda_var_amp;
	double lambda_var_mu;
	double lambda_var_sig;
        double sig_init;
	double ub_sig_init;
	double ub_sig;


	std::vector<std::vector<std::vector<double>>> cube;
      	std::vector<std::vector<std::vector<double>>> cube_mean;
	std::vector<std::vector<std::vector<double>>> fit_params;
	std::vector<std::vector<std::vector<double>>> grid_params;
	std::vector<std::vector<double>> std_cube;
	std::vector<std::vector<double>> std_map;
	std::vector<std::vector<double>> std_map_abs;
	std::vector<double> std_spect;
	std::vector<double> max_spect;
	std::vector<double> max_spect_norm;
       	std::vector<double> mean_spect;
	std::vector<double> guess_spect;

	std::vector<int> dim_data(3,0);
	std::vector<int> dim_cube(3,0);
  */	


/*
	std::vector<std::vector<std::vector<double>>> data;

        Parse file;

	file.get_binary_from_fits();

	file.get_vector_from_binary(data);

	file.multiresolution(1,data);
*/

        Parse file; //Créée data depuis le FITS en faisant une copie dans un fichier brut 

	file.multiresolution(5);	

	std::cout << std::max( 0, std::max(int(ceil( log(double(32))/log(2.))), int(ceil( log(double(32))/log(2.)))) ) << std::endl;

	return 0;
}
