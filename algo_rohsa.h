#ifndef DEF_ALGO_ROHSA
#define DEF_ALGO_ROHSA

#include "lbfgsb.h"
#include "hypercube.h"
#include "model.h"
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

class algo_rohsa
{
	public:

	algo_rohsa(model &M, const hypercube &Hypercube);	

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
	void init_bounds(model &M, std::vector<double> line, int n_gauss_local, std::vector<double> lb, std::vector<double> ub); // Conditions aux bord du spectre

	void mean_array(int power, std::vector<std::vector<std::vector<double>>> &mean_array_); //moyenne du tableau

	void init_spectrum(model &M, std::vector<double> line, std::vector<double> params); 
	//init spectre descente
	double model_function(int x, double a, double m, double s); //modèle

	int minloc(std::vector<double> tab); //argmin

	void minimize_spec(model &M, long n, long m, std::vector<double> x_v, std::vector<double> lb_v, std::vector<double> ub_v, std::vector<double> line_v); //LBFGS
	void old_minimize_spec(model &M, long n, long m, std::vector<double> x_v, std::vector<double> lb_v, std::vector<double> ub_v, std::vector<double> line_v); //LBFGS

	void myresidual(model &M, double params[], int line[], std::vector<double> residual);

	void tab_from_1Dvector_to_double(std::vector<double> vect);
	double myfunc_spec(std::vector<double> &residual);

	void mygrad_spec(model &M, double gradient[], std::vector<double> &residual, double params[]);

	void upgrade(model &M, std::vector<std::vector<std::vector<double>>> &cube, std::vector<std::vector<std::vector<double>>> params, int power);

//	void go_up_level(std::vector<std::vector<std::vector<double>>> &cube_params);

//	Computationnal tools
	private:

	std::vector<std::vector<double>> kernel;
	std::vector<int> dim_data; //inutile : file.dim_data
	int dim_x;
	int dim_y;
	int dim_v;
	hypercube file;

	int n_gauss_add; //EN DISCUTER AVEC ANTOINE

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


#endif