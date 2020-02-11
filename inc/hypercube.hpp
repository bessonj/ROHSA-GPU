#ifndef DEF_HYPERCUBE
#define DEF_HYPERCUBE

#include "model.hpp"
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

class hypercube
{
	public:

	hypercube();
	hypercube(model &M);
	hypercube(model &M,int indice_debut, int indice_fin);

	void display_cube(int rang);
	void display_data(int rang);
	void display(std::vector<std::vector<std::vector<double>>> &tab, int rang);
	void plot_line(std::vector<std::vector<std::vector<double>>> &params, int ind_x, int ind_y, int n_gauss_i);
	void display_result_and_data(std::vector<std::vector<std::vector<double>>> &params,int rang, int n_gauss_i);

	double model_function(int x, double a, double m, double s);
	void display_result(std::vector<std::vector<std::vector<double>>> &params, int rang, int n_gauss_i);

	int dim2nside(); //obtenir les dimensions 2^n
	void brute_show(const std::vector<std::vector<std::vector<double>>> &z, int depth, int length1, int length2);
	void multiresolution(int nside); 
	int get_binary_from_fits();
	void get_array_from_fits(model &M);
	void get_vector_from_binary(std::vector<std::vector<std::vector<double>>> &z);
	void show_data(); 
	std::vector<int> get_dim_data();
	std::vector<int> get_dim_cube();
	int get_nside() const;
	std::vector<std::vector<std::vector<double>>> use_dat_file(model &M);
	std::vector<std::vector<std::vector<double>>> reshape_up();
	std::vector<std::vector<std::vector<double>>> reshape_up(int borne_inf, int borne_sup);

	void write_into_binary(model &M);


	int indice_debut, indice_fin;
	std::vector<std::vector<std::vector<double>>> cube; //data format 2^n
	std::vector<std::vector<std::vector<double>>> data; //data brut
	int dim_data[3];
	int dim_cube[3];
	std::vector<int> dim_data_v;
	std::vector<int> dim_cube_v;
	int nside;

	std::string filename; 
};






#endif
