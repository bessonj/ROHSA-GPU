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

/**
 * @brief This class is about reading the fits file, transforming the data array and displaying the result in various ways.
 *
 *
 *
 *
 *  We can read a FITS or a DAT file, the result is then stored into a output file (binary or fits). The hypercube extracted is put into a larger hypercube of dimensions \f$ dim\_\nu \times 2^{n\_side} \times 2^{n\_side} \f$.
 * We can then rebuild the hypercube using the results, print the mean values of the gaussian parameters or print the gaussian parameters map. 
 * 
 *
 *
 * We read and write the FITS file using the library CCFits based on CFitsio.
 *
 *
 *
 * 
 *
 */

class hypercube
{
	public:

	hypercube();
	hypercube(parameters &M);
	hypercube(parameters &M,int indice_debut, int indice_fin); // assuming whole_data_in_cube = false (faster and better provided the dimensions are close)
	hypercube(parameters &M,int indice_debut, int indice_fin, bool whole_data_in_cube);
	hypercube(parameters &M,int indice_debut, int indice_fin, bool whole_data_in_cube, bool one_level);

	void display_cube(int rang);
	void display_data(int rang);
	void display(std::vector<std::vector<std::vector<double>>> &tab, int rang);
	void plot_line(std::vector<std::vector<std::vector<double>>> &params, int ind_x, int ind_y, int n_gauss_i);
	void plot_lines(std::vector<std::vector<std::vector<double>>> &params, std::vector<std::vector<std::vector<double>>> &cube_mean);
	void plot_multi_lines(std::vector<std::vector<std::vector<double>>> &params, std::vector<std::vector<std::vector<double>>> &cube_mean);
	void plot_multi_lines(std::vector<std::vector<std::vector<double>>> &params, std::vector<std::vector<std::vector<double>>> &cube_mean, std::string some_string);
	void display_result_and_data(std::vector<std::vector<std::vector<double>>> &params,int rang, int n_gauss_i, bool dat_or_not);
	void display_avec_et_sans_regu(std::vector<std::vector<std::vector<double>>> &params, int num_gauss, int num_par, int plot_numero);
	void display_2_gaussiennes(std::vector<std::vector<std::vector<double>>> &params,int rang, int n_gauss_i, int n1, int n2);
	void display_2_gaussiennes_par_par_par(std::vector<std::vector<std::vector<double>>> &params,int rang, int n_gauss_i, int n1, int n2);
	void mean_parameters(std::vector<std::vector<std::vector<double>>> &params, int num_gauss);
	void simple_plot_through_regu(std::vector<std::vector<std::vector<double>>> &params, int num_gauss, int num_par, int plot_numero);



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
	std::vector<std::vector<std::vector<double>>> reshape_up_for_last_level(int borne_inf, int borne_sup);

	void write_into_binary(parameters &M, std::vector<std::vector<std::vector<double>>> &grid_params);
	void get_from_file(std::vector<std::vector<std::vector<double>>> &file_out, int dim_0, int dim_1, int dim_2);
	void write_in_file(std::vector<std::vector<std::vector<double>>> &file_in);

	void write_vector_to_file(const std::vector<double>& myVector, std::string filename);
	std::vector<double> read_vector_from_file(std::string filename);

	template <typename T> void save_result(std::vector<std::vector<std::vector<T>>>&, parameters&);


	int indice_debut, indice_fin; //!< Only some spectral ranges of the hypercube are exploitable. We cut the hypercube, this will introduce an offset on the result values.
	std::vector<std::vector<std::vector<double>>> cube; //!< The hypercube "data" is centered into a larger hypercube "cube" for the purpose of multiresolution this hypercube is useful and its spatial dimensions are \f$  2^{n\_side} \times 2^{n\_side} \f$. Where \f$n\_side\f$ is computed by dim2nside(), it is the smallest power of 2 greater than the spatial dimensions. 
//	std::vector<std::vector<std::vector<double>>> data_not_reshaped;
	std::vector<std::vector<std::vector<double>>> data; //!< Hypercube array extracted from the fits file, its spectral range is changed according to indice_debut and indice_fin. 

	int dim_data[3];
	int dim_cube[3];
	std::vector<int> dim_data_v;
	std::vector<int> dim_cube_v;
	int nside;

	std::string filename; 
};






#endif
