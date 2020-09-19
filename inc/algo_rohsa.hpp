#ifndef DEF_ALGO_ROHSA
#define DEF_ALGO_ROHSA

#include "gradient.hpp"
#include "../L-BFGS-B-C/src/lbfgsb.h"
#include "hypercube.hpp"
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

class algo_rohsa
{
	public:

	algo_rohsa(model &M, hypercube &Hypercube); //constructeur

	void descente(model &M, std::vector<std::vector<std::vector<double>>> &grid_params, std::vector<std::vector<std::vector<double>>> &fit_params); //effectue la descente et/ou la régularisation
	void descente_sans_regu(model &M, std::vector<std::vector<std::vector<double>>> &grid_params, std::vector<std::vector<std::vector<double>>> &fit_params); //effectue la descente et/ou la régularisation

//	Computationnal tools
	void convolution_2D_mirror(const model &M, const std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &conv, int dim_y, int dim_x, int dim_k); //convolution 2D
	void ravel_2D(const std::vector<std::vector<double>> &map, std::vector<double> &vector, int dim_y, int dim_x);
	void ravel_3D(const std::vector<std::vector<std::vector<double>>> &cube, double vector[], int dim_v, int dim_y, int dim_x);
	void ravel_3D(const std::vector<std::vector<std::vector<double>>> &cube, std::vector<double> &vector, int dim_v, int dim_y, int dim_x);
	void ravel_3D_bis(const std::vector<std::vector<std::vector<double>>> &cube, double vector[], int dim_v, int dim_y, int dim_x);
	void ravel_3D_abs(const std::vector<std::vector<std::vector<double>>> &cube, const std::vector<std::vector<std::vector<double>>> &cube_abs, std::vector<double> &vector, int dim_v, int dim_y, int dim_x);
	void unravel_3D(const std::vector<double> &vector, std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x);
	void unravel_3D(double vector[], std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x);
	void unravel_3D_with_formula_transpose_xy(double vector[], std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x);
	void unravel_3D_T(double vector[], std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x);
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
	void init_bounds(model &M, std::vector<double> line, int n_gauss_local, std::vector<double> &lb, std::vector<double> &ub, bool _init); // Conditions aux bord du spectre

	void mean_array(int power, std::vector<std::vector<std::vector<double>>> &mean_array_); //moyenne du tableau

	void init_spectrum(model &M, std::vector<double> &line, std::vector<double> &params); 
	//init spectre descente
	double model_function(int x, double a, double m, double s); //modèle

	int minloc(std::vector<double> &tab); //argmin

	void minimize_spec(model &M, long n, long m, std::vector<double> &x_v, std::vector<double> &lb_v, int n_gauss_i,std::vector<double> &ub_v, std::vector<double> &line_v); //LBFGS

	 void minimize(model &M, long n, long m, std::vector<double> &x_v, std::vector<double> &lb_v, std::vector<double> &ub_v, std::vector<std::vector<std::vector<double>>> &cube, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig, int indice_x, int indice_y, int indice_v);

	void myresidual(double params[], double line[], std::vector<double> &residual, int n_gauss_i);
	void myresidual(std::vector<double> &params, std::vector<double> &line, std::vector<double> &residual, int n_gauss_i);

	void tab_from_1Dvector_to_double(std::vector<double> vect);
	double myfunc_spec(std::vector<double> &residual);

	void mygrad_spec(double gradient[], std::vector<double> &residual, double params[], int n_gauss_i);

	void upgrade(model &M, std::vector<std::vector<std::vector<double>>> &cube, std::vector<std::vector<std::vector<double>>> &params, int power);

	void go_up_level(std::vector<std::vector<std::vector<double>>> &fit_params);

	void set_stdmap(std::vector<std::vector<double>> &std_map, std::vector<std::vector<std::vector<double>>> &cube, int lb, int ub);
	void set_stdmap_transpose(std::vector<std::vector<double>> &std_map, std::vector<std::vector<std::vector<double>>> &cube, int lb, int ub);
	void update(model &M, std::vector<std::vector<std::vector<double>>> &cube, std::vector<std::vector<std::vector<double>>> &params, std::vector<std::vector<double>> &std_map, int indice_x, int indice_y, int indice_v,std::vector<double> &b_params);

	void f_g_cube_vector(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);
	void f_g_cube(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);
	void f_g_cube_sep(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);
	void f_g_cube_test(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);
	void f_g_cube_naive(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);
	void f_g_cube_fast(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);
	void f_g_cube_fast_without_regul(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);
	void f_g_cube_cuda_L(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);
	void f_g_cube_cuda_L_debug(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);
	void f_g_cube_cuda_L_2(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);
//	void f_g_cube_cuda_L_3(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);
	void f_g_cube_cuda_L_2_2Bverified(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);
	void f_g_cube_omp(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);
	void f_g_cube_omp_without_regul(model &M,double &f, double g[],  int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig);

	void reshape_down(std::vector<std::vector<std::vector<double>>> &tab1, std::vector<std::vector<std::vector<double>>>&tab2);





	std::vector<std::vector<std::vector<double>>> grid_params;
	std::vector<std::vector<std::vector<double>>> fit_params;
//	Computationnal tools
	private:

	std::vector<std::vector<double>> kernel;
	std::vector<int> dim_cube; //
	std::vector<int> dim_data; //

	int dim_x;
	int dim_y;
	int dim_v;
	hypercube file;

	double temps_f_g_cube;
	double temps_conv;
	double temps_deriv;
	double temps_tableaux;
	double temps_bfgs;
	double temps_update_beginning;
	double temps_;
	double temps_f_g_cube_tot;
	double temps_1_tot;
	double temps_2_tot;
	double temps_3_tot;
	double temps_ravel;
	double temps_tableau_update;

	int n_gauss_add; //EN DISCUTER AVEC ANTOINE

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
