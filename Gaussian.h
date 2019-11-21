//#ifndef DEF_GAUSSIAN
//#define DEF_GAUSSIAN

#include "lbfgsb.h"
#include "Parse.h"
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

class Gaussian
{
	public:

	Gaussian(const Parse &file1);	

//	Computationnal tools
	void convolution_2D_mirror(const std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &conv, int dim_y, int dim_x, int dim_k);
	void ravel_2D(const std::vector<std::vector<double>> &map, std::vector<double> &vector, int dim_y, int dim_x);
	void ravel_3D(const std::vector<std::vector<std::vector<double>>> &cube, std::vector<double> &vector, int dim_v, int dim_y, int dim_x);
	void ravel_3D_abs(const std::vector<std::vector<std::vector<double>>> &cube, const std::vector<std::vector<std::vector<double>>> &cube_abs, std::vector<double> &vector, int dim_v, int dim_y, int dim_x);
	void unravel_3D(const std::vector<double> &vector, std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x);
	void unravel_3D_abs(const std::vector<double> &vector, std::vector<std::vector<std::vector<double>>> &cube_abs,std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x);
	double Std(const std::vector<double> &array);
	double mean(const std::vector<double> &array);
	double std_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x);
	double max_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x);
	double mean_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x);
	void std_spectrum(int dim_x, int dim_y, int dim_v);
	void mean_spectrum(int dim_x, int dim_y, int dim_v);
	void max_spectrum(int dim_x, int dim_y, int dim_v);
	void max_spectrum_norm(int dim_x, int dim_y, int dim_v, double norm_value);
	void init_bounds(std::vector<double> line, int n_gauss_local, std::vector<double> lb, std::vector<double> ub, double ub_sig);

	void mean_array(int power, std::vector<std::vector<std::vector<double>>> &mean_array_);

	void init_spectrum(double ub_sig, std::vector<double> line, std::vector<double> params);
	double gaussian(int x, double a, double m, double s);

	int minloc(std::vector<double> tab);

	int minimize_spec(long n, long m, std::vector<double> x_v, std::vector<double> lb_v, std::vector<double> ub_v, std::vector<double> line_v);

	void tab_from_1Dvector_to_double(std::vector<double> vect);

	private:

	std::vector<std::vector<double>> kernel;
	std::vector<int> dim_data; //inutile : file.dim_data
	int dim_x;
	int dim_y;
	int dim_v;
	Parse file;

	int n_gauss_add; //EN DISCUTER AVEC ANTOINE
	

//	parameters
	std::string filename;
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

	std::vector<double> std_spect, mean_spect, max_spect, max_spect_norm;
	std::vector<std::vector<std::vector<double>>> grid_params;
	std::vector<std::vector<std::vector<double>>> fit_params;
	

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



