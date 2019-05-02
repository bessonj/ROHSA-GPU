#include "Gaussian.h"


Gaussian::Gaussian()
{
	std::string txt, egal;

        std::ifstream fichier("parameters.txt", std::ios::in);  // on ouvre en lecture
 
        if(fichier)  // si l'ouverture a fonctionné
        {
		fichier >> txt >> egal >> filename;
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
		fichier >> txt >> egal >> noise;
		fichier >> txt >> egal >> regul;
		fichier >> txt >> egal >> descent;
		fichier >> txt >> egal >> lstd;
		fichier >> txt >> egal >> ustd;
		fichier >> txt >> egal >> iprint;
		fichier >> txt >> egal >> iprint_init;
		fichier >> txt >> egal >> save_grid;



                fichier.close();
        }
        else
                std::cerr << "Impossible d'ouvrir le fichier !" << std::endl;
/*
	dim_data = file.get_dim_data();
	dim_x = dim_data[0];
	dim_y = dim_data[1];
	dim_v = dim_data[2];

	dim_data2dim_cube();

	nside = 5;
*/
}

