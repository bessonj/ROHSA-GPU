#include "model.hpp"
#include <omp.h>

model::model()
{
	n_gauss_add = 0;

	std::string txt, egal;
        std::ifstream fichier("parameters.txt", std::ios::in);  // on ouvre en lecture
 
        if(fichier)  // si l'ouverture a fonctionné
        {
		fichier >> txt >> egal >> filename_dat;
		fichier >> txt >> egal >> filename_fits;
		fichier >> txt >> egal >> file_type_dat_check;
		fichier >> txt >> egal >> file_type_fits_check;
		fichier >> txt >> egal >> slice_index_min;
		fichier >> txt >> egal >> slice_index_max;
		fichier >> txt >> egal >> fileout;
		fichier >> txt >> egal >> filename_noise;
		fichier >> txt >> egal >> n_gauss;
		fichier >> txt >> egal >> lambda_amp;
		fichier >> txt >> egal >> lambda_mu;
		fichier >> txt >> egal >> lambda_sig;
		fichier >> txt >> egal >> lambda_var_amp;
		fichier >> txt >> egal >> lambda_var_mu;
		fichier >> txt >> egal >> lambda_var_sig;
		fichier >> txt >> egal >> amp_fact_init;
		fichier >> txt >> egal >> sig_init;
		fichier >> txt >> egal >> init_option;
		fichier >> txt >> egal >> maxiter_init;
		fichier >> txt >> egal >> maxiter;
		fichier >> txt >> egal >> m;
		fichier >> txt >> egal >> check_noise;
		fichier >> txt >> egal >> check_regul;
		fichier >> txt >> egal >> check_descent;
		fichier >> txt >> egal >> lstd;
		fichier >> txt >> egal >> ustd;
		fichier >> txt >> egal >> iprint;
		fichier >> txt >> egal >> iprint_init;
		fichier >> txt >> egal >> check_save_grid;
		fichier >> txt >> egal >> ub_sig;
		fichier >> txt >> egal >> lb_sig;
		fichier >> txt >> egal >> ub_sig_init;
		fichier >> txt >> egal >> lb_sig_init;
		if(file_type_dat_check == "true")
			file_type_dat = true;
		else 
			file_type_dat = false;
		if(file_type_fits_check == "true")
			file_type_fits = true;
		else 
			file_type_fits = false;

		if(check_save_grid == "true")
			save_grid = true;
		else 
			save_grid = false;		
		if(check_noise == "true")
			noise = true;
		else
			noise = false;
		if(check_regul == "true")
			regul = true;
		else
			regul = false;
		if(check_descent == "true")
			descent = true;
		else
			descent = false;

                fichier.close();
        }
        else
                std::cerr << "Impossible d'ouvrir le fichier !" << std::endl;

	std::vector<double> slice(3,0.);
	kernel.vector::push_back(slice);
	kernel.vector::push_back(slice);
	kernel.vector::push_back(slice);
	slice.clear();

	kernel[0][0] = 0.;
	kernel[0][1] = -0.25;
	kernel[0][2] = 0.;
	kernel[1][0] = -0.25;
	kernel[1][1] = 1.;
	kernel[1][2] = -0.25;
	kernel[2][0] = 0.;
	kernel[2][1] = -0.25;
	kernel[2][2] = 0.;

}


void model::write_model_in_binary(){

//	

}
