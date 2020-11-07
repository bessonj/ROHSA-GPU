#ifndef DEF_HYPERCUBE
#define DEF_HYPERCUBE

#include "parameters.hpp"
#include <iostream>
#include <stdio.h>
#include <cmath>
//#include <math.h>
#include <string>
#include <fstream>
#include <valarray>
#include <CCfits/CCfits>
#include <vector>

/// This class is about reading the fits file, transforming the data array and displaying the result in various ways.
/// 
///  

class hypercube
{
	public:

	hypercube();
	hypercube(parameters &M);
	hypercube(parameters &M,int indice_debut, int indice_fin); // assuming whole_data_in_cube = false (faster and better provided the dimensions are close)
	hypercube(parameters &M,int indice_debut, int indice_fin, bool whole_data_in_cube);

	void display_cube(int rang);
	void display_data(int rang);
	void display(std::vector<std::vector<std::vector<double>>> &tab, int rang);
	void plot_line(std::vector<std::vector<std::vector<double>>> &params, int ind_x, int ind_y, int n_gauss_i);
	void display_result_and_data(std::vector<std::vector<std::vector<double>>> &params,int rang, int n_gauss_i, bool dat_or_not);
	void display_avec_et_sans_regu(std::vector<std::vector<std::vector<double>>> &params, int num_gauss, int num_par, int plot_numero);
	void display_2_gaussiennes(std::vector<std::vector<std::vector<double>>> &params,int rang, int n_gauss_i, int n1, int n2);
	void display_2_gaussiennes_par_par_par(std::vector<std::vector<std::vector<double>>> &params,int rang, int n_gauss_i, int n1, int n2);
	void mean_parameters(std::vector<std::vector<std::vector<double>>> &params, int num_gauss);




	double model_function(int x, double a, double m, double s);
	void display_result(std::vector<std::vector<std::vector<double>>> &params, int rang, int n_gauss_i);

	int dim2nside(); //obtenir les dimensions 2^n
	void brute_show(const std::vector<std::vector<std::vector<double>>> &z, int depth, int length1, int length2);
	void multiresolution(int nside); 
	int get_binary_from_fits();
	void get_array_from_fits(parameters &M);
	void get_vector_from_binary(std::vector<std::vector<std::vector<double>>> &z);
	void show_data(); 
	std::vector<int> get_dim_data();
	std::vector<int> get_dim_cube();
	int get_nside() const;
	std::vector<std::vector<std::vector<double>>> use_dat_file(parameters &M);
	std::vector<std::vector<std::vector<double>>> reshape_up();
	std::vector<std::vector<std::vector<double>>> reshape_up(int borne_inf, int borne_sup);

	void write_into_binary(parameters &M, std::vector<std::vector<std::vector<double>>> &grid_params);


	int indice_debut, indice_fin;
	std::vector<std::vector<std::vector<double>>> cube; //data format 2^n and width given by user (parameters.txt)
//	std::vector<std::vector<std::vector<double>>> data_not_reshaped; //data without width given by user (parameters.txt) might blow up memory
	std::vector<std::vector<std::vector<double>>> data; //data with width given by user (parameters.txt)

	int dim_data[3];
	int dim_cube[3];
	std::vector<int> dim_data_v;
	std::vector<int> dim_cube_v;
	int nside;

	std::string filename; 
};






#endif