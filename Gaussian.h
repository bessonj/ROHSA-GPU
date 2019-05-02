#ifndef DEF_GAUSSIAN
#define DEF_GAUSSIAN

#include "Parse.h"

// mettre des const à la fin des déclarations si on ne modifie pas l'objet i.e. les attributs

class Gaussian
{
	public:

	Gaussian();	
//	void dim_data2dim_cube();


	private:

	Parse file;

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
	bool noise;
	bool regul;
	bool descent;
	int lstd;
	int ustd;
	int iprint;
	int iprint_init;
	bool save_grid;
/*
	int n_gauss_add;
	int nside;
	int n;
	int power;
	double ub_sig_init;
	double ub_sig;

	std::vector<int> dim_data;
	int dim_x;
	int dim_y;
	int dim_v;
	std::vector<int> dim_cube; 

*/


};

#endif


