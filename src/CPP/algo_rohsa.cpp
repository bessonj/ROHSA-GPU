#include "algo_rohsa.hpp"
#include <array>
#include "gradient.hpp"
#include "convolutions.hpp"
#include "f_g_cube_gpu.hpp"



algo_rohsa::algo_rohsa(parameters &M, hypercube &Hypercube)
{

// to be deleted
	std::cout<<"hypercube[0][0][0] = "<<Hypercube.cube[0][0][0]<<std::endl;
	std::cout<<"hypercube[1][0][0] = "<<Hypercube.cube[1][0][0]<<std::endl;
	std::cout<<"hypercube[0][1][0] = "<<Hypercube.cube[0][1][0]<<std::endl;
	std::cout<<"hypercube[0][0][1] = "<<Hypercube.cube[0][0][1]<<std::endl;
//	exit(0);
/*
	for(int u = 0; u<490; u++){
//		if (Hypercube.data[0][0][u]<-0.016 && Hypercube.data[0][0][u]>-0.017){
			std::cout<<"hypercube[0][0]["<<u<<"] = "<<Hypercube.data[0][0][u]<<std::endl;
//		}
	}
*/

	this->file = Hypercube; //The hypercube is not modified afterwards

//  Dimensions of data and cube
	this->dim_cube = Hypercube.get_dim_cube();
	this->dim_data = Hypercube.get_dim_data();

//	Dimensions of the cube /!\ dim_x, dim_y, dim_v stand for the spatial and spectral dimensions of the cube
	this->dim_x = dim_cube[0];
	this->dim_y = dim_cube[1];
	this->dim_v = dim_cube[2];
	std_spectrum(dim_data[0], dim_data[1], dim_data[2]); //oublier
	mean_spectrum(dim_data[0], dim_data[1], dim_data[2]);
	max_spectrum(dim_data[0], dim_data[1], dim_data[2]); //oublier

	//compute the maximum of the mean spectrum
	double max_mean_spect = *std::max_element(mean_spect.begin(), mean_spect.end());

	max_spectrum_norm(dim_data[0], dim_data[1], dim_data[2], max_mean_spect);


	std::cout << " descent : "<< M.descent << std::endl;

	std::vector<std::vector<std::vector<double>>> grid_params, fit_params;

// can't define the proper variable in the loop 
	if(M.descent){
	std::vector<std::vector<std::vector<double>>> grid_params_(3*(M.n_gauss+(file.nside*M.n_gauss_add)), std::vector<std::vector<double>>(dim_data[1], std::vector<double>(dim_data[0],0.)));
	std::vector<std::vector<std::vector<double>>> fit_params_(3*(M.n_gauss+(file.nside*M.n_gauss_add)), std::vector<std::vector<double>>(1, std::vector<double>(1,0.)));
	grid_params=grid_params_;
	fit_params=fit_params_;
	}
	else{
	std::vector<std::vector<std::vector<double>>> grid_params_(3*(M.n_gauss+M.n_gauss_add), std::vector<std::vector<double>>(dim_data[1], std::vector<double>(dim_data[0],0.)));
	grid_params=grid_params_;
	}

			
//		std::cout << "test fit_params : "<<fit_params[0][0][0]<<std::endl;
		
	if(M.descent)
	{
		std::cout<<"DÉBUT DE DESCENTE"<<std::endl;
		descente(M, grid_params, fit_params);
	}

	std::cout << "params.size() : " << fit_params.size() << " , " << fit_params[0].size() << " , " << fit_params[0][0].size() <<  std::endl;
	std::cout << "params.size() : " << grid_params.size() << " , " << grid_params[0].size() << " , " << grid_params[0][0].size() <<  std::endl;
/*
	for(int i(0); i<grid_params.size(); i++) {
		for(int j(0); j<grid_params[0].size(); j++) {
			for(int k(0); k<grid_params[0][0].size(); k++) {
				std::cout<<"Après setulb, params["<<i<<"]["<<j<<"]["<<k<<"] = "<<grid_params[i][j][k]<<std::endl;
			}
		}
	}
*/

	this->file.write_into_binary(M, this->grid_params);

}

void algo_rohsa::descente(parameters &M, std::vector<std::vector<std::vector<double>>> &grid_params, std::vector<std::vector<std::vector<double>>> &fit_params){
	
	std::vector<double> b_params(M.n_gauss,0.);
	temps_f_g_cube = 0.; 
	temps_conv = 0.;
	temps_deriv = 0.;
	temps_tableaux = 0.;
	temps_bfgs = 0.;
	temps_setulb = 0.;
	temps_transfert = 0.;
	temps_update_beginning = 0.;
	temps_tableau_update = 0.;
	temps_mirroirs = 0.;
	for(int i=0;i<M.n_gauss; i++){
		fit_params[0+3*i][0][0] = 0.;
		fit_params[1+3*i][0][0] = 1.;
		fit_params[2+3*i][0][0] = 1.;
	}
	std::cout << "fit_params.size() : " << fit_params.size() << " , " << fit_params[0].size() << " , " << fit_params[0][0].size() <<  std::endl;

	std::cout << "grid_params.size() : "<< grid_params.size() << " , " << grid_params[0].size()  << " , " << grid_params[0][0].size() << std::endl;

	double temps1_descente = omp_get_wtime();

	std::vector<double> fit_params_flat(fit_params.size(),0.); //used below

	double temps_multiresol=0.;

	double temps_init_spectrum=0.;
	double temps_upgrade=0.;
	double temps_update_pp=0.;
	double temps_update_dp=0.;
	double temps_go_up_level=0.;
	double temps_reshape_down=0.;
	double temps_std_map_pp=0.;
	double temps_std_map_dp=0.;
	double temps_dernier_niveau = 0.;

	temps_ravel=0.;
	int n;
//		#pragma omp parallel private(n) shared(temps_upgrade, temps_multiresol, temps_init, temps_mean_array,M, fit_params_flat,file)
//		{
//		#pragma omp for
	double temps1_before_nside = omp_get_wtime();

	for(n=0; n<file.nside; n++)
	{
		double temps1_init_spectrum = omp_get_wtime();

		int power(pow(2,n));

		std::cout << " power = " << power << std::endl;

		std::vector<std::vector<std::vector<double>>> cube_mean(power, std::vector<std::vector<double>>(power,std::vector<double>(dim_v,1.)));

		mean_array(power, cube_mean);

		std::vector<double> cube_mean_flat(cube_mean[0][0].size());

		if (n==0) {
			for(int e(0); e<cube_mean[0][0].size(); e++) {
				cube_mean_flat[e] = cube_mean[0][0][e]; //cache ok
				}

			for(int e(0); e<fit_params_flat.size(); e++) {
				fit_params_flat[e] = fit_params[e][0][0]; //cache   USELESS SINCE NO ITERATION OCCURED BEFORE
				}



			//assume option "mean"
			std::cout<<"Init mean spectrum"<<std::endl;
			
			init_spectrum(M, cube_mean_flat, fit_params_flat);

//				init_spectrum(M, cube_mean_flat, std_spect); //option spectre
//				init_spectrum(M, cube_mean_flat, max_spect); //option max spectre
//				init_spectrum(M, cube_mean_flat, max_spect_norm); //option norme spectre


			for(int i(0); i<M.n_gauss; i++) {
					b_params[i]= fit_params_flat[2+3*i];
//					std::cout<<"b_params["<<i<<"] = "<<b_params[i]<<std::endl;
				}

			//we recover fit_params from its 1D version since we can't do fit_params[.][1][1] in C/C++
			for(int e(0); e<fit_params_flat.size(); e++) {
				fit_params[e][0][0] = fit_params_flat[e]; //cache
				}
			}
		//	exit(0);
		
		double temps2_init_spectrum = omp_get_wtime();
		temps_init_spectrum+= temps2_init_spectrum - temps1_init_spectrum;
		double temps1_upgrade = omp_get_wtime();
			if(M.regul==false) {
				double temps1_upgrade = omp_get_wtime();
				for(int e(0); e<fit_params.size(); e++) {
					fit_params[e][0][0]=fit_params_flat[e];
					grid_params[e][0][0] = fit_params[e][0][0];
				}
				upgrade(M ,cube_mean, fit_params, power);
				double temps2_upgrade = omp_get_wtime();
				temps_upgrade+=temps2_upgrade-temps1_upgrade;

			} else if(M.regul) {
				if (n==0){
					double temps1_upgrade = omp_get_wtime();
					upgrade(M ,cube_mean, fit_params, power);
					double temps2_upgrade = omp_get_wtime();
					temps_upgrade+=temps2_upgrade-temps1_upgrade;
				}
				if (n>0 and n<file.nside){
					std::vector<std::vector<double>> std_map(power, std::vector<double>(power,0.));
					double temps_std_map1=omp_get_wtime();
					if (M.noise){
						//reshape_noise_up(indice_debut, indice_fin);
						//mean_map()	
					}
					
					else if (M.noise==false){
						set_stdmap_transpose(std_map, cube_mean, M.lstd, M.ustd);

//						set_stdmap(std_map, cube_mean, M.lstd, M.ustd); //?
					}
					double temps_std_map2=omp_get_wtime();
					temps_std_map_pp+=temps_std_map2-temps_std_map1;

					double temps1_update_pp=omp_get_wtime();

					update(M, cube_mean, fit_params, std_map, power, power, dim_v, b_params);

/*
	printf("fit_params[0][0][0] = %f \n",fit_params[0][0][0] );
	exit(0);
*/
/*
for(int i = 0; i<(3*M.n_gauss*indice_x*indice_y); i++){
	printf("fit_[%d] = %f \n",i,g[i] );
}
*/

					double temps2_update_pp=omp_get_wtime();
					temps_update_pp += temps2_update_pp-temps1_update_pp;

					if (M.n_gauss_add != 0){
						//Add new Gaussian if one reduced chi square > 1  
					}
					mean_parameters(fit_params);
				}
			}

	
			if (M.save_grid){
				//save grid in file
			}

	double temps_go_up_level1=omp_get_wtime();
			go_up_level(fit_params);
			this->fit_params = fit_params; //updating the model class
	double temps_go_up_level2=omp_get_wtime();
	temps_go_up_level=temps_go_up_level2-temps_go_up_level1;

	}

/*
	this->grid_params = fit_params;
	return;
*/

/*
	std::cout<<"AFFICHAGE POUR LA RECHERCHE D'ERREUR"<<std::endl;

		for(int j(0); j<fit_params[0].size(); j++) {
			for(int k(0); k<fit_params[0][0].size(); k++) {
				std::cout<<"-> fit_params["<<0<<"]["<<j<<"]["<<k<<"] = "<<fit_params[0][j][k]<<std::endl;
			}
		}
		exit(0);	
	*/
	double temps2_before_nside = omp_get_wtime();



	std::cout<<"            Milieu descente             "<<std::endl;
	std::cout<<"                                "<<std::endl;
	std::cout<<"Temps écoulé jusque ici dans la descente : "<<omp_get_wtime() - temps1_descente <<std::endl;
	std::cout<<"	-> Temps de calcul init_spectrum : "<< temps_init_spectrum <<std::endl;
	std::cout<<"	-> Temps de calcul upgrade (update 1D) : "<< temps_upgrade <<std::endl;
	std::cout<<"	-> Temps de calcul std_map : "<< temps_std_map_pp <<std::endl;
	std::cout<<"	-> Temps de calcul update (update 1->n-1) : "<< temps_update_pp <<std::endl;
	std::cout<<"	-> Temps de calcul go_up_level (grille k->k+1) : "<<temps_go_up_level <<std::endl;
	std::cout<<"                                "<<std::endl;
	std::cout<<"                                "<<std::endl;
	std::cout<<"                                "<<std::endl;
	std::cout<<"                                "<<std::endl;
	std::cout<<"            début détails             "<<std::endl;
	std::cout<<"                                "<<std::endl;
	std::cout<< "temps d'exécution dF/dB : "<<temps_f_g_cube<<std::endl;
	std::cout<< "Temps d'exécution convolution : " << temps_conv <<std::endl;
	std::cout<< "Temps d'exécution deriv : " << temps_deriv  <<std::endl;
	std::cout<< "Temps d'exécution ravel_3D : " << temps_ravel <<std::endl;
	std::cout<< "Temps d'exécution tableaux : " << temps_tableaux <<std::endl;
	std::cout<<"                                "<<std::endl;
	std::cout<<"           fin détails             "<<std::endl;
	std::cout<<"                                "<<std::endl;


	std::vector<std::vector<double>> std_map(this->dim_data[1], std::vector<double>(this->dim_data[0],0.));

	double temps_dernier_niveau1 = omp_get_wtime();

	//nouvelle place de reshape_down
	int offset_w = (this->dim_cube[0]-this->dim_data[0])/2;
	int offset_h = (this->dim_cube[1]-this->dim_data[1])/2;

std::cout<<"Taille fit_params : "<<fit_params.size()<<" , "<<fit_params[0].size()<<" , "<<fit_params[0][0].size()<<std::endl;
std::cout<<"Taille grid_params : "<<grid_params.size()<<" , "<<grid_params[0].size()<<" , "<<grid_params[0][0].size()<<std::endl;

//ancienne place de reshape_down	
	double temps_reshape_down2 = omp_get_wtime();
	reshape_down(fit_params, grid_params);
	double temps_reshape_down1 = omp_get_wtime();
	temps_reshape_down2 += temps_reshape_down2-temps_reshape_down1;

std::cout<<"Après reshape_down"<<std::endl;
	this->grid_params = grid_params;


//std::cout<<"Taille fit_params : "<<fit_params.size()<<" , "<<fit_params[0].size()<<" , "<<fit_params[0][0].size()<<std::endl;

////PRINT GRID
/*
	for(int j(0); j<8; j++) {
		for(int k(0); k<8; k++) {
			std::cout<<"--> grid_params["<<0<<"]["<<j<<"]["<<k<<"] = "<<grid_params[0][j][k]<<std::endl;
		}
	}
*/

/*
std::cout<<"this->file.data = "<<this->file.data[0][0][0]<<std::endl;
std::cout<<"this->dim_data[0] = "<<this->dim_data[0]<<std::endl;
std::cout<<"this->dim_data[1] = "<<this->dim_data[1]<<std::endl;
std::cout<<"this->dim_data[2] = "<<this->dim_data[2]<<std::endl;
*/
//exit(0);
	double temps_std_map1=omp_get_wtime();
	if(M.noise){
		//std_map=std_data;
	} else {

		set_stdmap(std_map, this->file.data, M.lstd, M.ustd);
	}
	double temps_std_map2=omp_get_wtime();
	temps_std_map_dp+=temps_std_map2-temps_std_map1;

/*
		std::cout<<"--> std_map[0][0] = "<<std_map[0][0]<<std::endl;
		std::cout<<"--> std_map[1][0] = "<<std_map[1][0]<<std::endl;
		std::cout<<"--> std_map[0][1] = "<<std_map[0][1]<<std::endl;
*/
	double temps_update_dp1 = omp_get_wtime();

	if(M.regul){
		std::cout<<"Updating last level"<<std::endl;
		// repère recherche pb %
		update(M, this->file.data, grid_params, std_map, this->dim_data[0], this->dim_data[1], this->dim_v, b_params);
		//modification update
	}

	double temps_update_dp2 = omp_get_wtime();
	temps_update_dp +=temps_update_dp2-temps_update_dp1;

//ancienne place de reshape_down	
	this->grid_params = grid_params;
	int comptage = 600;


////PRINT GRID
/*
		for(int j(0); j<grid_params[0].size(); j++) {
			for(int k(0); k<grid_params[0][0].size(); k++) {
					std::cout<<"--> grid_params["<<0<<"]["<<j<<"]["<<k<<"] = "<<grid_params[0][j][k]<<std::endl;
			}
		}
*/

		double temps2_descente = omp_get_wtime();
//		std::cout<<"fit_params_flat["<<0<<"]= "<<"  vérif:  "<<fit_params[0][0][0]<<std::endl;
		double temps_dernier_niveau2 = omp_get_wtime();
		temps_dernier_niveau+=temps_dernier_niveau2-temps_dernier_niveau1;


	std::cout<<"Temps de descente : "<<temps2_descente - temps1_descente <<std::endl;
	std::cout<<"Temps de calcul niveaux 1 -> n-1 : "<<temps2_before_nside - temps1_before_nside <<std::endl;
	std::cout<<"	-> Temps de calcul init_spectrum : "<< temps_init_spectrum <<std::endl;
	std::cout<<"	-> Temps de calcul upgrade (update 1D) : "<< temps_upgrade <<std::endl;
	std::cout<<"	-> Temps de calcul std_map : "<< temps_std_map_pp <<std::endl;
	std::cout<<"	-> Temps de calcul update (update 1->n-1) : "<< temps_update_pp <<std::endl;
	std::cout<<"	-> Temps de calcul go_up_level (grille k->k+1) : "<<temps_go_up_level <<std::endl;
	std::cout<<"Temps de calcul reshape_down (n-1 -> n)"<<temps_reshape_down <<std::endl;
	std::cout<<"Temps de calcul niveau n : "<< temps_dernier_niveau <<std::endl;
	std::cout<<"	-> Temps de calcul std_map : "<< temps_std_map_dp <<std::endl;
	std::cout<<"	-> Temps de calcul update : "<< temps_update_dp <<std::endl;


	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;
	std::cout<<" "<<std::endl;
	std::cout<<"                                "<<std::endl;
	std::cout<<"            début détails             "<<std::endl;
	std::cout<<"                                "<<std::endl;
	std::cout<<"temps d'exécution résidu et f : "<<temps_f_g_cube<<std::endl;
	std::cout<<"temps d'exécution setulb : "<<temps_setulb<<std::endl;
	std::cout<< "Temps d'exécution convolution : " << temps_conv <<std::endl;
	std::cout<< "	-> Temps d'exécution routine mirroir : " << temps_mirroirs <<std::endl;
	std::cout<< "	-> Temps de transfert : " << temps_transfert <<std::endl;
	std::cout<< "Temps d'exécution deriv : " << temps_deriv  <<std::endl;
	std::cout<< "Temps d'exécution ravel_3D : " << temps_ravel <<std::endl;
	std::cout<< "Temps d'exécution tableaux avant gradient : " << temps_tableaux <<std::endl;
	std::cout<< "Temps d'exécution tableaux dans update : " << temps_tableau_update <<std::endl;
	std::cout<<"                                "<<std::endl;
	std::cout<<"           fin détails             "<<std::endl;
	std::cout<<"                                "<<std::endl;
}

void algo_rohsa::reshape_down(std::vector<std::vector<std::vector<double>>> &tab1, std::vector<std::vector<std::vector<double>>>&tab2)
{
	int dim_tab1[3], dim_tab2[3];
	dim_tab1[0]=tab1.size();
	dim_tab1[1]=tab1[0].size();
	dim_tab1[2]=tab1[0][0].size();
	dim_tab2[0]=tab2.size();
	dim_tab2[1]=tab2[0].size();
	dim_tab2[2]=tab2[0][0].size();

	int offset_w = (dim_tab1[1]-dim_tab2[1])/2;
	int offset_h = (dim_tab1[2]-dim_tab2[2])/2;



	for(int i(0); i< dim_tab1[0]; i++)
	{
		for(int j(0); j<dim_tab2[1]; j++)
		{
			for(int k(0); k<dim_tab2[2]; k++)
			{
//				std::cout<<"i = "<<i << " , j = "<< j<< " , k = "<<k<<std::endl;
				tab2[i][j][k] = tab1[i][offset_w+j][offset_h+k];
			}
		}
	}

}

void algo_rohsa::update(parameters &M, std::vector<std::vector<std::vector<double>>> &cube_avgd_or_data, std::vector<std::vector<std::vector<double>>> &params, std::vector<std::vector<double>> &std_map, int indice_x, int indice_y, int indice_v, std::vector<double> &b_params) {

//printf("b_params[0] = %f \n",b_params[0] );
//printf("beta \n");

	//cube flattened for array operations in f_g_cube_cuda
	//WARNING : free(cube_flattened) at the end of the function
	double* cube_flattened = NULL;
	size_t size_cube = indice_x*indice_y*indice_v*sizeof(double); 
	cube_flattened = (double*)malloc(size_cube);

	for(int i=0; i<indice_x; i++) {
		for(int j=0; j<indice_y; j++) {
			for(int k=0; k<indice_v; k++) {
				cube_flattened[k*indice_x*indice_y+j*indice_x+i] = cube_avgd_or_data[i][j][k];
			}
		}
	}

//	this->cube_or_dat_flattened = cube_flattened; 

std::cout<<"Taille params : "<<params.size()<<" , "<<params[0].size()<<" , "<<params[0][0].size()<<std::endl;
std::cout<<"Taille std_map : "<<std_map.size()<<" , "<<std_map[0].size()<<std::endl;

//EXTRAIT 
/*
		for(int j(0); j<int(std::min(8,int(params[0].size()))); j++) {
			for(int k(0); k<int(std::min(8,int(params[0][0].size()))); k++) {
					std::cout<<"---> params["<<0<<"]["<<j<<"]["<<k<<"] = "<<params[0][j][k]<<std::endl;
			}
		}
*/

std::cout<<"this->file.data = "<<this->file.data[0][0][0]<<std::endl;
std::cout<<"this->dim_data[0] = "<<this->dim_data[0]<<std::endl;
std::cout<<"this->dim_data[1] = "<<this->dim_data[1]<<std::endl;
std::cout<<"this->dim_data[2] = "<<this->dim_data[2]<<std::endl;

	std::cout<<"DÉBUT DE UPDATE"<<std::endl;
	int n_beta = (3*M.n_gauss * indice_y * indice_x) +M.n_gauss;

std::cout<<"indice_x = "<<indice_x<<std::endl;
std::cout<<"indice_y = "<<indice_y<<std::endl;
std::cout<<"n_gauss = "<<M.n_gauss<<std::endl;

	std::vector<double> lb(n_beta,0.);
	std::vector<double> ub(n_beta,0.);
	std::vector<double> beta(n_beta,0.);

	std::vector<std::vector<std::vector<double>>> lb_3D(3*M.n_gauss, std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
	std::cout << "lb_3D.size() : " << lb_3D.size() << " , " << lb_3D[0].size() << " , " << lb_3D[0][0].size() <<  std::endl;



	std::vector<std::vector<double>> image_amp(indice_y, std::vector<double>(indice_x,0.));
	std::vector<std::vector<double>> image_mu(indice_y, std::vector<double>(indice_x,0.));
	std::vector<std::vector<double>> image_sig(indice_y, std::vector<double>(indice_x,0.));

	std::vector<double> mean_amp(M.n_gauss,0.);
	std::vector<double> mean_mu(M.n_gauss,0.);
	std::vector<double> mean_sig(M.n_gauss,0.);

	std::vector<double> ravel_amp(indice_y*indice_x,0.);
	std::vector<double> ravel_mu(indice_y*indice_x,0.);
	std::vector<double> ravel_sig(indice_y*indice_x,0.);

	std::vector<double> cube_flat(cube_avgd_or_data[0][0].size(),0.);
	std::vector<double> lb_3D_flat(lb_3D.size(),0.);

	std::vector<std::vector<std::vector<double>>> ub_3D(3*M.n_gauss, std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
//	std::vector<std::vector<std::vector<double>>> ub_3D;//(3*M.n_gauss, std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
//	ub_3D = lb_3D; //otherwise we get an error : "corrupted double-linked list"
std::cout << "ub_3D.size() : " << ub_3D.size() << " , " << ub_3D[0].size() << " , " << ub_3D[0][0].size() <<  std::endl;

	std::vector<double> ub_3D_flat(ub_3D.size(),0.);

	double temps1_tableau_update = omp_get_wtime();
//	std::cout<<"ERREUR 0"<<std::endl;
	std::cout << "indice_x = " << indice_x <<std::endl;
	std::cout << "indice_y = " << indice_y <<std::endl;
	std::cout << "cube.size() : " << cube_avgd_or_data.size() << " , " << cube_avgd_or_data[0].size() << " , " << cube_avgd_or_data[0][0].size() <<  std::endl;
/*
if(cube_avgd_or_data[0].size() == 35){
	exit(0);
}
*/
	for(int j=0; j<indice_x; j++) {
		for(int i=0; i<indice_y; i++) {
			for(int p=0; p<cube_flat.size(); p++){
				cube_flat[p]=cube_avgd_or_data[j][i][p];
			}

//	std::cout<<"ERREUR 1"<<std::endl;
			for(int p=0; p<3*M.n_gauss; p++){
				lb_3D_flat[p]=lb_3D[p][i][j];
				ub_3D_flat[p]=ub_3D[p][i][j];
			}
//	std::cout<<"ERREUR 2"<<std::endravel_3D(l;
			init_bounds(M, cube_flat, M.n_gauss, lb_3D_flat, ub_3D_flat, false); //bool _init = false 
//	std::cout<<"ERREUR 3"<<std::endl;
			for(int p=0; p<3*M.n_gauss; p++){
				lb_3D[p][i][j]=lb_3D_flat[p]; // the indices have been inverted
				ub_3D[p][i][j]=ub_3D_flat[p]; //
			}
		}
	}
	//ravel_3D(lb_3D, lb, indice_x, indice_y, 3*M.n_gauss); 
	ravel_3D(lb_3D, lb, 3*M.n_gauss, indice_y, indice_x);//ancien
	//ravel_3D(ub_3D, ub, indice_x, indice_y, 3*M.n_gauss);
	ravel_3D(ub_3D, ub, 3*M.n_gauss, indice_y, indice_x); //ancien
	//ravel_3D(params, beta, 3*M.n_gauss, indice_y, indice_x);
	ravel_3D(params, beta, 3*M.n_gauss, indice_y, indice_x); //ancien

//printf("b_params[0] = %f \n",b_params[0] );
//printf("beta \n");

	for(int i=0; i<M.n_gauss; i++){
		lb[n_beta-M.n_gauss+i] = M.lb_sig;
		ub[n_beta-M.n_gauss+i] = M.ub_sig;
		beta[n_beta-M.n_gauss+i] = b_params[i];
	}

//printf("beta[0] = %f \n",beta[0] );
//printf("beta \n");

//	std::cout<<"FIN DE BETA"<<std::endl;

	temps_tableau_update += omp_get_wtime() - temps1_tableau_update;
//	std::cout<< "AVANT MINIMIZE" <<std::endl;

/*
	for(int i=0; i<M.n_gauss; i++){
		std::cout<< "b_params["<<i<<"] = "<<b_params[i] <<std::endl;
	}
	exit(0);
*/

/*
	for(int p(0); p<3*M.n_gauss; p++) {
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				std::cout<<"params["<<p<<"]["<<i<<"]["<<j<<"] = "<<params[p][i][j]<<std::endl;
			}
		}
	}
	exit(0);
*/
	std::cout<<"AVANT MINIMIZE"<<std::endl;
	//erreur ici
	minimize(M, n_beta, M.m, beta, lb, ub, cube_avgd_or_data, std_map, mean_amp, mean_mu, mean_sig, indice_x, indice_y, indice_v, cube_flattened); 

	std::cout<< "--> beta[0] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! = "<<beta[0] <<std::endl;

	double temps2_tableau_update = omp_get_wtime();
	unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

/*
	std::cout<< "params[0][0][0] = "<<params[0][0][0] <<std::endl;
	std::cout<< "params[1][0][0] = "<<params[1][0][0] <<std::endl;
	std::cout<< "params[0][1][0] = "<<params[0][1][0] <<std::endl;
	std::cout<< "params[0][0][1] = "<<params[0][0][1] <<std::endl;
//exit(0);
*/
	for(int i=0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

/*
	std::cout<< "APRÈS MINIMIZE" <<std::endl;
	for(int i=0; i<M.n_gauss; i++){
		std::cout<< "b_params["<<i<<"] = "<<b_params[i] <<std::endl;
	}
*/
//exit(0);

	temps_tableau_update += omp_get_wtime() - temps2_tableau_update;

	free(cube_flattened);
}

void algo_rohsa::set_stdmap(std::vector<std::vector<double>> &std_map, std::vector<std::vector<std::vector<double>>> &cube_or_data, int lb, int ub){
	std::vector<double> line(ub-lb+1,0.);
	int dim[3];
	dim[2]=cube_or_data[0][0].size();
	dim[1]=cube_or_data[0].size();
	dim[0]=cube_or_data.size();
	for(int j=0; j<dim[1]; j++){
		for(int i=0; i<dim[0]; i++){
			for(int p=0; p<line.size(); p++){
				line[p] = cube_or_data[i][j][p+lb];
			}
			std_map[j][i] = Std(line);
		}
	}
}

//Je n'ai pas modifié le nom, mais ce n'est pas transposé !!!! [i][j] ==> normal [j][i] ==> transposé
void algo_rohsa::set_stdmap_transpose(std::vector<std::vector<double>> &std_map, std::vector<std::vector<std::vector<double>>> &cube, int lb, int ub){
	std::vector<double> line(ub-lb+1,0.);


	int dim[3];
	dim[2]=cube[0][0].size();
	dim[1]=cube[0].size();
	dim[0]=cube.size();

/*
	printf("taille line = %d \n", ub-lb);
	std::cout<<"line.size() = "<<line.size()<<std::endl;
	std::cout<<"ub = "<<ub<<std::endl;
	std::cout<<"lb = "<<lb<<std::endl;
	std::cout<<"cube[0][0][0] = "<<cube[0][0][0]<<std::endl;
	std::cout<<"cube[0][0][1] = "<<cube[0][0][1]<<std::endl;
	std::cout<<"cube[0][0][2] = "<<cube[0][0][2]<<std::endl;
	std::cout<<"cube[0][0][3] = "<<cube[0][0][3]<<std::endl;
	std::cout<<"cube[0][0][4] = "<<cube[0][0][4]<<std::endl;
	std::cout<<"cube[0][0][5] = "<<cube[0][0][5]<<std::endl;
	std::cout<<"cube[0][0][6] = "<<cube[0][0][6]<<std::endl;
*/
//	std::cout<<"dim_0 = "<<dim[0]<<std::endl;
//	std::cout<<"dim_1 = "<<dim[1]<<std::endl;
//	exit(0);
	for(int j=0; j<dim[1]; j++){
		for(int i=0; i<dim[0]; i++){
			for(int p=0; p<= ub-lb; p++){
				line[p] = cube[i][j][p+lb];
			}
			std_map[j][i] = Std(line);
//		printf("Std(line) = %f \n",Std(line));
		}
	}

/*
	for(int o = 0; o<ub-lb+1; o++){
		printf("line[%d] = %f \n",o, line[o]);
	}
	for(int j=0; j<dim[1]; j++){
		for(int i=0; i<dim[0]; i++){
			printf("std_map[%d][%d] = %f \n",j,i, std_map[j][i]);
		}
	}
	exit(0);
*/
}

void algo_rohsa::f_g_cube_fast(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig){
	std::vector<std::vector<std::vector<double>>> deriv(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
	std::vector<std::vector<std::vector<double>>> g_3D(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));

	std::vector<std::vector<std::vector<double>>> residual(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_v,0.)));

	std::vector<std::vector<std::vector<double>>> params(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
	std::vector<double> b_params(M.n_gauss,0.);

	std::vector<std::vector<double>> conv_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_sig(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_sig(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_sig(indice_y,std::vector<double>(indice_x, 0.));

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;

	for(int i = 0; i< n; i++){
		g[i]=0.;
	}
	f=0.;

	unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

	for(int j=0; j<indice_x; j++){
		for(int i=0; i<indice_y; i++){
			std::vector<double> residual_1D(indice_v,0.);
			std::vector<double> params_flat(params.size(),0.);
			std::vector<double> cube_flat(cube[0][0].size(),0.);

			for (int p=0;p<params_flat.size();p++){
				params_flat[p]=params[p][i][j];
			}

//			std::cout<< "--> cube_flat.size() = "<<cube_flat.size() <<std::endl;
			for (int p=0;p<cube_flat.size();p++){
				cube_flat[p]=cube[j][i][p];
			}

			myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);

			for (int p=0;p<residual_1D.size();p++){
				residual[j][i][p]=residual_1D[p];
			}

			if(std_map[i][j]>0.){
				f += myfunc_spec(residual_1D)*1/pow(std_map[i][j],2.); //std_map est arrondie... 
			}
		}
	}

/*
	printf("std_map[0][0] = %f \n",std_map[0][0]);
	printf("std_map[0][1] = %f \n",std_map[0][1]);
	printf("std_map[1][0] = %f \n",std_map[1][0]);
	printf("std_map[1][1] = %f \n",std_map[1][1]);
	printf("-->f = %f\n",f);
	exit(0);
*/

//§§

//	std::cout<<"->f = "<<f<<std::endl;
/*
	for(int j=0; j<indice_y; j++){
		for(int i = 0; i<indice_x;i++){
		std::cout<<"std_map["<<i<<"]["<<j<<"] = "<<std_map[i][j]<<std::endl;
		}
	}
*/
//	exit(0);

//	std::cout<< "AVANT GRADIENT+CONVO" <<std::endl;

	for(int i=0; i<M.n_gauss; i++){
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){
				image_amp[p][q]=params[0+3*i][p][q];
				image_mu[p][q]=params[1+3*i][p][q];
				image_sig[p][q]=params[2+3*i][p][q];
			}
		}

//		std::cout<< "AVANT CONVO" <<std::endl;

		convolution_2D_mirror(M, image_amp, conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror(M, image_mu, conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror(M, image_sig, conv_sig, indice_y, indice_x,3);
	
		convolution_2D_mirror(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);

//		std::cout<< "AVANT SOMME F" <<std::endl;
		for(int l=0; l<indice_x; l++){
			for(int j=0; j<indice_y; j++){
				f+= 0.5*M.lambda_amp*pow(conv_amp[j][l],2);
				f+= 0.5*M.lambda_mu*pow(conv_mu[j][l],2);
				f+= 0.5*M.lambda_sig*pow(conv_sig[j][l],2)+0.5*M.lambda_var_sig*pow(image_sig[j][l]-b_params[i],2);

				g[n_beta-M.n_gauss+i] += M.lambda_var_sig*(b_params[i]-image_sig[j][l]);
			
		int k;
		double v_1=0.,v_2=0.,v_3=0.;

//		std::cout<< "AVANT GRADIENT BIDOUILLÉ" <<std::endl;

		if(std_map[j][l]>0.){
			for(k=0; k<indice_v; k++){
				v_1 += exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
				v_2 += params[3*i][j][l]*( double(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
				v_3 += params[3*i][j][l]*pow( double(k+1) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
			}
		}
		deriv[0+3*i][j][l] += v_1 + M.lambda_amp*conv_conv_amp[j][l];
		deriv[1+3*i][j][l] += v_2 + M.lambda_mu*conv_conv_mu[j][l];
		deriv[2+3*i][j][l] += v_3 + M.lambda_sig*conv_conv_sig[j][l]+M.lambda_var_sig*(image_sig[j][l]-b_params[i]);



/*
	printf("--->M.lambda_amp = %f\n",M.lambda_amp);
	printf("--->M.lambda_mu = %f\n",M.lambda_mu);
	printf("--->M.lambda_sig = %f\n",M.lambda_sig);

	printf("--->conv_conv_amp = %f\n",conv_conv_amp[j][l]);
	printf("--->conv_conv_mu = %f\n",conv_conv_mu[j][l]);
	printf("--->conv_conv_sig = %f\n",conv_conv_sig[j][l]);

	printf("--->deriv = %f\n",deriv[0+3*i][j][l]);
	printf("--->deriv = %f\n",deriv[1+3*i][j][l]);
	printf("--->deriv = %f\n",deriv[2+3*i][j][l]);
	exit(0);	
*/
			}
		}
		

	}
//		std::cout<< "AVANT TABLEAUX FIN" <<std::endl;

	ravel_3D(deriv, g, 3*M.n_gauss, indice_y, indice_x);
	//	std::cout<<"g["<<i<<"] = "<<g[i]<<std::endl;
//	std::cout<<"g["<<i<<"] = "<<g[i]<<std::endl;

/*
	for(int i = 0; i<n_beta;i++)
	{
		printf("g[%i] = %f\n",i,g[i]);
	}
	printf("--->f = %f\n",f);
	exit(0);
*/
//	exit(0);


	}


void algo_rohsa::f_g_cube_fast_without_regul(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig){

	std::vector<std::vector<std::vector<double>>> deriv(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
	std::vector<std::vector<std::vector<double>>> g_3D(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));

	std::vector<std::vector<std::vector<double>>> residual(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_v,0.)));

	std::vector<std::vector<std::vector<double>>> params(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
	std::vector<double> b_params(M.n_gauss,0.);

	std::vector<std::vector<double>> conv_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_sig(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_sig(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_sig(indice_y,std::vector<double>(indice_x, 0.));

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;//+M.n_gauss;

	for(int i = 0; i< n; i++){
		g[i]=0.;
	}
	f=0.;

	unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

	for(int j=0; j<indice_x; j++){
		for(int i=0; i<indice_y; i++){
			std::vector<double> residual_1D(indice_v,0.);
			std::vector<double> params_flat(params.size(),0.);
			std::vector<double> cube_flat(cube[0][0].size(),0.);
			for (int p=0;p<params_flat.size();p++){
				params_flat[p]=params[p][i][j];
			}
			for (int p=0;p<cube_flat.size();p++){
				cube_flat[p]=cube[i][j][p];
			}
			myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);
			for (int p=0;p<residual_1D.size();p++){
				residual[j][i][p]=residual_1D[p];
			}
			if(std_map[i][j]>0.){
				f += myfunc_spec(residual_1D)*1/pow(std_map[i][j],2.); //std_map est arrondie... 
			}
		}
	}
//§§
//	printf("->f = %f\n",f);
//	std::cout<<"->f = "<<f<<std::endl;
/*
	for(int j=0; j<indice_y; j++){
		for(int i = 0; i<indice_x;i++){
		std::cout<<"std_map["<<i<<"]["<<j<<"] = "<<std_map[i][j]<<std::endl;
		}
	}
*/
//	exit(0);
int k,i,l,j;
#pragma omp parallel private(i,l,j,k) shared(std_map,deriv,params,M,indice_v, residual,indice_y,indice_x)
{
#pragma omp for
	for(i=0; i<M.n_gauss; i++){
		for(l=0; l<indice_x; l++){
			for(j=0; j<indice_y; j++){
				double v_1=0.,v_2=0.,v_3=0.;

			if(std_map[j][l]>0.){
	//	    #pragma omp parallel shared(M, params, residual, std_map, indice_v,i,j,l)
	//	    {
	//        #pragma omp for private(k)
			for(k=0; k<indice_v; k++){
				v_1 += exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
	//printf("--->curseur = %e\n", std_map[j][l] );//residual[l][j][k]  );
				v_2 +=  params[3*i][j][l]*( double(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
		
				v_3 += params[3*i][j][l]*pow( double(k+1) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
						}
		}
		deriv[0+3*i][j][l]+=v_1;
		deriv[1+3*i][j][l]+=v_2;
		deriv[2+3*i][j][l]+=v_3;
//		}
			}
		}
	}
	}
	
	ravel_3D(deriv, g, 3*M.n_gauss, indice_y, indice_x);
	//	std::cout<<"g["<<i<<"] = "<<g[i]<<std::endl;
//	std::cout<<"g["<<i<<"] = "<<g[i]<<std::endl;

/*
	for(int i = 0; i<n_beta;i++)
	{
		printf("g[%i] = %f\n",i,g[i]);
	}
	printf("--->f = %f\n",f);
*/
//	exit(0);
//jeudi
	}


//OpenMP for f_g_cube
void algo_rohsa::f_g_cube_vector(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig){

//std::cout<<"Début f_g_cube"<<std::endl;


std::vector<std::vector<std::vector<double>>> dR_over_dB(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));

//décommenter en bas

std::vector<std::vector<std::vector<std::vector<double>>>> dF_over_dB(3*M.n_gauss,std::vector<std::vector<std::vector<double>>>( indice_v,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.))));
std::vector<std::vector<std::vector<double>>> deriv(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<std::vector<std::vector<double>>> g_3D(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));

std::vector<std::vector<std::vector<double>>> residual(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_v,0.)));

std::vector<std::vector<std::vector<double>>> params(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<double> b_params(M.n_gauss,0.);

std::vector<std::vector<double>> conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_sig(indice_y,std::vector<double>(indice_x, 0.));

int i,k,j,l,p;

int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;//+M.n_gauss;
double temps1_tableaux = omp_get_wtime();
for(int i = 0; i< n; i++){
	g[i]=0.;
}
f=0.;

unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);
for(int i = 0; i<M.n_gauss; i++){
	b_params[i]=beta[n_beta-M.n_gauss+i];
}

for(int j=0; j<indice_x; j++){
	for(int i=0; i<indice_y; i++){
		std::vector<double> residual_1D(indice_v,0.);
		std::vector<double> params_flat(params.size(),0.);
		std::vector<double> cube_flat(cube[0][0].size(),0.);
		for (int p=0;p<params_flat.size();p++){
			params_flat[p]=params[p][i][j];
		}
		for (int p=0;p<cube_flat.size();p++){
			cube_flat[p]=cube[j][i][p];
		}
		myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);
		for (int p=0;p<residual_1D.size();p++){
			residual[j][i][p]=residual_1D[p];
		}
		if(std_map[i][j]>0.){
			f += myfunc_spec(residual_1D)*1/pow(std_map[i][j],2.); //std_map est arrondie... 
		}
	}
}

double temps2_tableaux = omp_get_wtime();

double temps1_dF_dB = omp_get_wtime();

/*
//ANCIEN CODE (à tester) sans optim cache
for(int i=0; i<M.n_gauss; i++){
	for(int k=0; k<indice_v; k++){
		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){
				dF_over_dB[0+3*i][k][j][l] += exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

				dF_over_dB[1+3*i][k][j][l] +=  params[3*i][j][l]*( double(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

				dF_over_dB[2+3*i][k][j][l] += params[3*i][j][l]*pow( double(k+1) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );
			}
		}
	}
}
*/
// // // CALCUL DU GRADIENT

//transfert ->

//appel kernel 
//gradient_kernel<<<,>>>(

//transfert <-


//OMP
/*
#pragma omp parallel private(i,k,j) shared(dF_over_dB,params,M,indice_v,indice_y,indice_x)
{
#pragma omp for
for(int i=0; i<M.n_gauss; i++){
	for(k=0; k<indice_v; k++){
		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){
				dF_over_dB[0+3*i][k][j][l] += exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

				dF_over_dB[1+3*i][k][j][l] +=  params[3*i][j][l]*( double(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * 
									exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

				dF_over_dB[2+3*i][k][j][l] += params[3*i][j][l]*pow( double(k+1) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * 
									exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );
			}
		}
	}
}
}

*/
//repère1

#pragma omp parallel private(i,k) shared(dF_over_dB,params,M,indice_v,indice_y,indice_x) //num_threads(1) //
{
#pragma omp for
for(int i=0; i<M.n_gauss; i++){
        for(k=0; k<indice_v; k++){
                for(int j=0; j<indice_y; j++){
                        for(int l=0; l<indice_x; l++){

				dF_over_dB[0+3*i][k][j][l] += exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

				dF_over_dB[1+3*i][k][j][l] +=  params[3*i][j][l]*( double(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * 
									exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

				dF_over_dB[2+3*i][k][j][l] += params[3*i][j][l]*pow( double(k+1) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * 
									exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

                        }
                }
        }
}
}

double temps2_dF_dB = omp_get_wtime();

double temps1_deriv = omp_get_wtime();
#pragma omp parallel private(k,j) shared(deriv,dF_over_dB,M,indice_v,indice_y,indice_x,std_map,residual)
{
#pragma omp for
for(k=0; k<indice_v; k++){
	for(j=0; j<indice_x; j++){
		for(int i=0; i<indice_y; i++){
			for(int l=0; l<3*M.n_gauss; l++){
				if(std_map[i][j]>0.){
					deriv[l][i][j]+=  dF_over_dB[l][k][i][j]*residual[j][i][k]/pow(std_map[i][j],2);
				}
			}
		}
	}
}
}




double temps2_deriv = omp_get_wtime();

double temps1_conv = omp_get_wtime();

for(int k=0; k<M.n_gauss; k++){

	for(int p=0; p<indice_y; p++){
		for(int q=0; q<indice_x; q++){
			image_amp[p][q]=params[0+3*k][p][q];
			image_mu[p][q]=params[1+3*k][p][q];
			image_sig[p][q]=params[2+3*k][p][q];
		}
	}

	convolution_2D_mirror(M, image_amp, conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, image_mu, conv_mu, indice_y, indice_x,3);
	convolution_2D_mirror(M, image_sig, conv_sig, indice_y, indice_x,3);

	convolution_2D_mirror(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);


	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			f+= 0.5*M.lambda_amp*pow(conv_amp[i][j],2);
			f+= 0.5*M.lambda_mu*pow(conv_mu[i][j],2);
			f+= 0.5*M.lambda_sig*pow(conv_sig[i][j],2) + 0.5*M.lambda_var_sig*pow(image_sig[i][j]-b_params[k],2);

			dR_over_dB[0+3*k][i][j] = M.lambda_amp*conv_conv_amp[i][j];
			dR_over_dB[1+3*k][i][j] = M.lambda_mu*conv_conv_mu[i][j];
			dR_over_dB[2+3*k][i][j] = M.lambda_sig*conv_conv_sig[i][j]+M.lambda_var_sig*(image_sig[i][j]-b_params[k]);

			g[n_beta-M.n_gauss+k] += M.lambda_var_sig*(b_params[k]-image_sig[i][j]);
		}
	}
	}

	for(int l=0; l<3*M.n_gauss; l++){
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				g_3D[l][i][j] = deriv[l][i][j] + dR_over_dB[l][i][j];
			}
		}
	}
	ravel_3D(g_3D, g, 3*M.n_gauss, indice_y, indice_x);

	double temps2_conv = omp_get_wtime();

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;

	temps_f_g_cube += temps2_dF_dB - temps1_dF_dB;

}

void algo_rohsa::f_g_cube_old_archive(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig){


	std::vector<std::vector<std::vector<double>>> deriv(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
	std::vector<std::vector<std::vector<double>>> g_3D(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));

	std::vector<std::vector<std::vector<double>>> residual(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_v,0.)));

	std::vector<std::vector<std::vector<double>>> params(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
	std::vector<double> b_params(M.n_gauss,0.);

	std::vector<std::vector<double>> conv_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_sig(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_sig(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_sig(indice_y,std::vector<double>(indice_x, 0.));

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;//+M.n_gauss;

	for(int i = 0; i< n; i++){
		g[i]=0.;
	}
	f=0.;

	unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

	for(int j=0; j<indice_x; j++){
		for(int i=0; i<indice_y; i++){
			std::vector<double> residual_1D(indice_v,0.);
			std::vector<double> params_flat(params.size(),0.);
			std::vector<double> cube_flat(cube[0][0].size(),0.);
			for (int p=0;p<params_flat.size();p++){
				params_flat[p]=params[p][i][j];
			}
			for (int p=0;p<cube_flat.size();p++){
				cube_flat[p]=cube[i][j][p];
			}
			myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);
			for (int p=0;p<residual_1D.size();p++){
				residual[j][i][p]=residual_1D[p];
			}
			if(std_map[i][j]>0.){
				f+=myfunc_spec(residual_1D)/pow(std_map[i][j],2.); //std_map est arrondie... 
			}
		}
	}
	
	for(int i=0; i<M.n_gauss; i++){
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){
				image_amp[p][q]=params[0+3*i][p][q];
				image_mu[p][q]=params[1+3*i][p][q];
				image_sig[p][q]=params[2+3*i][p][q];
			}
		}
	
		convolution_2D_mirror(M, image_amp, conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror(M, image_mu, conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror(M, image_sig, conv_sig, indice_y, indice_x,3);
	
		convolution_2D_mirror(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);
		
		for(int l=0; l<indice_x; l++){
			for(int j=0; j<indice_y; j++){
				f+= 0.5*M.lambda_amp*pow(conv_amp[j][l],2);
				f+= 0.5*M.lambda_mu*pow(conv_mu[j][l],2);
				f+= 0.5*M.lambda_sig*pow(conv_sig[j][l],2)+0.5*M.lambda_var_sig*pow(image_sig[j][l]-b_params[i],2);
	
				g[n_beta-M.n_gauss+i] += M.lambda_var_sig*(b_params[i]-image_sig[j][l]);
	
				for(int k=0; k<indice_v; k++){
					if(std_map[j][l]>0.){
					deriv[0+3*i][j][l] += exp(-pow( double(k)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
	
					deriv[1+3*i][j][l] +=  params[3*i][j][l]*( double(k) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( double(k)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
	
					deriv[2+3*i][j][l] += params[3*i][j][l]*pow( double(k) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( double(k)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
					}
				}
				deriv[0+3*i][j][l] += M.lambda_amp*conv_conv_amp[j][l];
				deriv[1+3*i][j][l] += M.lambda_mu*conv_conv_mu[j][l];
				deriv[2+3*i][j][l] += M.lambda_sig*conv_conv_sig[j][l]+M.lambda_var_sig*(image_sig[j][l]-b_params[i]);
			}
		}
	}
	
	ravel_3D(deriv, g, 3*M.n_gauss, indice_y, indice_x);
}

//test optims OpenMP for f_g_cube
void algo_rohsa::f_g_cube_test(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig){

//std::cout<<"Début f_g_cube_test"<<std::endl;

std::vector<std::vector<std::vector<double>>> dR_over_dB(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(3*M.n_gauss,0.)));
//std::vector<std::vector<std::vector<double>>> dR_over_dB_ok(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
//°
//décommenter en bas
std::vector<std::vector<std::vector<std::vector<double>>>> dF_over_dB(indice_v,std::vector<std::vector<std::vector<double>>>( indice_y,std::vector<std::vector<double>>(indice_x, std::vector<double>(3*M.n_gauss,0.))));
//std::vector<std::vector<std::vector<std::vector<double>>>> dF_over_dB_ok(3*M.n_gauss,std::vector<std::vector<std::vector<double>>>( indice_v,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.))));
std::vector<std::vector<std::vector<double>>> deriv(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(3*M.n_gauss,0.)));
//std::vector<std::vector<std::vector<double>>> deriv_ok(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<std::vector<std::vector<double>>> g_3D(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(3*M.n_gauss,0.)));
//std::vector<std::vector<std::vector<double>>> g_3D_ok(3*M.n_gauss,std::vector<std::vector<double>>(indice_y,std::vector<double>(indice_x,0.)));
//deriv x y g
std::vector<std::vector<std::vector<double>>> residual(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_v,0.)));
std::vector<std::vector<std::vector<double>>> residual_T(indice_v,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));

std::vector<std::vector<std::vector<double>>> params(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<std::vector<std::vector<double>>> params_T(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(3*M.n_gauss,0.)));
std::vector<double> b_params(M.n_gauss,0.);

std::vector<std::vector<double>> conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_sig(indice_y,std::vector<double>(indice_x, 0.));

int n_beta = (3*M.n_gauss*indice_x*indice_y);//+M.n_gauss;

int i,k,j,l,p;

for(int i = 0; i< n; i++){
	g[i]=0.;
}
f=0.;

double temps1_ravel = omp_get_wtime();

//extracting one params_T for the gradient and another for calculations below

unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);
// returns params[g][y][x]
/*
for(int k=0; k<3*M.n_gauss; k++) {
 for(int j=0; j<indice_y; j++) {
  for(int i=0; i<indice_x; i++) {
   params_T[k][j][i]=beta[k*indice_y*indice_x+j*indice_x+i];
   //deriv x y g
   }
  }
 }
*/

unravel_3D_T(beta, params_T, 3*M.n_gauss, indice_y, indice_x);
//unravel_3D_with_formula_transpose_xy(beta, params_T, indice_y, indice_x, 3*M.n_gauss);

double temps2_ravel = omp_get_wtime();
temps_ravel+=temps2_ravel-temps1_ravel;

/*for(int i = 0; i<M.n_gauss; i++){
	b_params[i]=beta[n_beta-M.n_gauss+i];
}*/
//cout.precision(dbl::max_digits10);


//#pragma omp parallel private(j,i) shared(cube,params,std_map,residual,indice_x,indice_y,indice_v,M,f)
//{
//#pragma omp for
double temps1_tableaux = omp_get_wtime();
for(j=0; j<indice_x; j++){
	for(i=0; i<indice_y; i++){
		std::vector<double> residual_1D(indice_v,0.);
		std::vector<double> params_flat(params.size(),0.);
		std::vector<double> cube_flat(cube[0][0].size(),0.);
		for (p=0;p<params_flat.size();p++){
			params_flat[p]=params[p][i][j];
		}
		for (p=0;p<cube_flat.size();p++){
			cube_flat[p]=cube[i][j][p];
		}
		myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);
		for (p=0;p<residual_1D.size();p++){
			residual_T[p][i][j]=residual_1D[p];
		}
		if(std_map[i][j]>0.){
			f+=myfunc_spec(residual_1D)/pow(std_map[i][j],2.); //std_map est arrondie... 
		}
	}
}
/*
for(j=0; j<indice_x; j++){
	for(i=0; i<indice_y; i++){
		std::vector<double> residual_1D(indice_v,0.);
		std::vector<double> params_flat(params.size(),0.);
		std::vector<double> cube_flat(cube[0][0].size(),0.);
		for (p=0;p<params_flat.size();p++){
			params_flat[p]=params[p][i][j];
		}
		for (p=0;p<cube_flat.size();p++){
			cube_flat[p]=cube[i][j][p];
		}
		myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);
		for (p=0;p<residual_1D.size();p++){
			residual[j][i][p]=residual_1D[p];
		}
		if(std_map[i][j]>0.){
			f+=myfunc_spec(residual_1D)/pow(std_map[i][j],2.); //std_map est arrondie... 
		}
	}
}
*/

double temps2_tableaux = omp_get_wtime();

double temps1_dF_dB = omp_get_wtime();
/*
double par0 = 0.;
double par1 = 0.;
double par1_k = 0.;
double par2 = 0.;
double par2_pow = 0.;
*/
// v y x 3*i+.
//std::cout<<"DEBUG 1"<<std::endl;
#pragma omp parallel private(k,l,j,i) shared(dF_over_dB,params_T,M,indice_v,indice_y,indice_x)
{
#pragma omp for
for(int k=0; k<indice_v; k++){
	for(int l=0; l<indice_y; l++){
		for(int j=0; j<indice_x; j++){
			for(int i=0; i<M.n_gauss; i++){
				//ojqpsfdijpsqojdfpoidspoidfpojqspodij sdjfpjqspodfijiopqs jdpofijojfoiq sdoipf oqisdhfophjqsopidf opis dofi poqs dfop qspoidf jopiqjsdopif jopqisd fop Âsopid fpoij qsdf
				double par0 = params_T[l][j][3*i+0];
				double par1 = params_T[l][j][3*i+1];
				double par2 = params_T[l][j][3*i+2];

				double par1_k = par1 = double(k+1) - par1;
				double par2_pow = 1/(2*pow(par2,2.));
				//dF_over_dB v y x g

				dF_over_dB[k][l][j][3*i+0] += exp( -pow( par1_k ,2.)*par2_pow );
				dF_over_dB[k][l][j][3*i+1] += par0*par1_k*par2_pow*2 * exp(-pow( par1_k,2.)*par2_pow );
				dF_over_dB[k][l][j][3*i+2] += par0*pow( par1_k, 2.)/(pow(par2,3.)) * exp(-pow( par1_k,2.)*par2_pow );

/*
				dF_over_dB[k][l][j][3*i+0] += exp(-pow( double(k+1)-params_T[j][l][1+3*i],2.)/(2*pow(params_T[j][l][2+3*i],2.)) );
				dF_over_dB[k][l][j][3*i+1] += params_T[j][l][3*i]*( double(k+1) - params_T[j][l][1+3*i])/pow(params_T[j][l][2+3*i],2.) * 
							exp(-pow( double(k+1)-params_T[j][l][1+3*i],2.)/(2*pow(params_T[j][l][2+3*i],2.)) );
				dF_over_dB[k][l][j][3*i+2] += params_T[j][l][3*i]*pow( double(k+1) - params_T[j][l][1+3*i], 2.)/(pow(params_T[j][l][2+3*i],3.)) * 
							exp(-pow( double(k+1)-params_T[j][l][1+3*i],2.)/(2*pow(params_T[j][l][2+3*i],2.)) );
*/
		 
			//dF_over_dB_ok g v y x

			}
		}
	}
}
}

double temps2_dF_dB = omp_get_wtime();

double temps1_deriv = omp_get_wtime();

for(k=0; k<indice_v; k++){
	for(j=0; j<indice_x; j++){
      	for(i=0; i<indice_y; i++){
	        if(std_map[i][j]>0.){
         	for(l=0; l<3*M.n_gauss; l++){
				deriv[j][i][l]+=  dF_over_dB[k][i][j][l]*residual_T[k][i][j]/pow(std_map[i][j],2);
					//deriv x y g

				}
			}
		}
	}
}
//}


double temps2_deriv = omp_get_wtime();

int q;

double temps1_conv = omp_get_wtime();

for(int k=0; k<M.n_gauss; k++){

	for(p=0; p<indice_y; p++){
		for(q=0; q<indice_x; q++){
			image_amp[p][q]=params[0+3*k][p][q];
			image_mu[p][q]=params[1+3*k][p][q];
			image_sig[p][q]=params[2+3*k][p][q];
		}
	}

	convolution_2D_mirror(M, image_amp, conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, image_mu, conv_mu, indice_y, indice_x,3);
	convolution_2D_mirror(M, image_sig, conv_sig, indice_y, indice_x,3);

	convolution_2D_mirror(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);



	for(int j=0; i<indice_y; j++){
		for(int i=0; j<indice_x; i++){
			f+= 0.5*M.lambda_amp*pow(conv_amp[i][j],2) + 0.5*M.lambda_var_amp*pow(image_amp[i][j]-mean_amp[k],2);
			f+= 0.5*M.lambda_mu*pow(conv_mu[i][j],2) + 0.5*M.lambda_var_mu*pow(image_mu[i][j]-mean_mu[k],2);
			f+= 0.5*M.lambda_sig*pow(conv_sig[i][j],2) + 0.5*M.lambda_var_sig*pow(image_sig[i][j]-mean_sig[k],2);

/*
			dR_over_dB_ok[0+3*k][i][j] = M.lambda_amp*conv_conv_amp[i][j]+M.lambda_var_amp*(image_amp[i][j]-mean_amp[k]);
			dR_over_dB_ok[1+3*k][i][j] = M.lambda_mu*conv_conv_mu[i][j]+M.lambda_var_mu*(image_mu[i][j]-mean_mu[k]);
			dR_over_dB_ok[2+3*k][i][j] = M.lambda_sig*conv_conv_sig[i][j]+M.lambda_var_sig*(image_sig[i][j]-mean_sig[k]);
*/
			dR_over_dB[j][i][0+3*k] = M.lambda_amp*conv_conv_amp[i][j]+M.lambda_var_amp*(image_amp[i][j]-mean_amp[k]);
			dR_over_dB[j][i][1+3*k] = M.lambda_mu*conv_conv_mu[i][j]+M.lambda_var_mu*(image_mu[i][j]-mean_mu[k]);
			dR_over_dB[j][i][2+3*k] = M.lambda_sig*conv_conv_sig[i][j]+M.lambda_var_sig*(image_sig[i][j]-mean_sig[k]);
		}
	}

} 
/*
#pragma omp parallel private(i,j) shared(g_3D,deriv,dR_over_dB,M,indice_y,indice_x)
{
#pragma omp for
*/
for(int j=0; j<indice_x; j++){
	for(int i=0; i<indice_y; i++){
		for(int l=0; l<3*M.n_gauss; l++){
//				g_3D_ok[l][i][j] = deriv_ok[l][i][j] + dR_over_dB_ok[l][i][j];
//				g_3D[j][i][l] = deriv[j][i][l] + dR_over_dB[j][i][l];
				g[l+i*3*M.n_gauss+j*indice_y*3*M.n_gauss] = deriv[j][i][l] + dR_over_dB[j][i][l];
			}
		}
	}
//}

//	ravel_3D(g_3D_ok, g, 3*M.n_gauss, indice_y, indice_x);
//	ravel_3D(g_3D, g_test, 3*M.n_gauss, indice_y, indice_x);
//	ravel_3D(g_3D, g_test, indice_x, indice_y, 3*M.n_gauss);
/*
#pragma omp parallel private(i,j,k) shared(g,g_3D,indice_x,indice_y,M)
{
#pragma omp for
for(int i=0; i<indice_x; i++) {
 for(int j=0; j<indice_y; j++) {
  for(int k=0; k<3*M.n_gauss; k++) {
   g[k+j*3*M.n_gauss+i*indice_y*3*M.n_gauss] = g_3D[i][j][k];
   //deriv x y g
   }
  }
 }
}
*/


	double temps2_conv = omp_get_wtime();

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;

	temps_f_g_cube += temps2_dF_dB - temps1_dF_dB;
}


//test optims OpenMP for f_g_cube
void algo_rohsa::f_g_cube_sep(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig) 
{
	std::vector<std::vector<std::vector<double>>> deriv(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(3*M.n_gauss,0.)));
	
	std::vector<std::vector<std::vector<double>>> residual(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_v,0.)));

	std::vector<std::vector<std::vector<double>>> params(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
	std::vector<double> b_params(M.n_gauss,0.);

	std::vector<std::vector<double>> conv_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_sig(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_sig(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_sig(indice_y,std::vector<double>(indice_x, 0.));

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;//+M.n_gauss;

	for(int i = 0; i< n; i++){
		g[i]=0.;
	}
	f=0.;

	unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

	for(int j=0; j<indice_x; j++){
		for(int i=0; i<indice_y; i++){
			std::vector<double> residual_1D(indice_v,0.);
			std::vector<double> params_flat(params.size(),0.);
			std::vector<double> cube_flat(cube[0][0].size(),0.);
			for (int p=0;p<params_flat.size();p++){
				params_flat[p]=params[p][i][j];
			}
			for (int p=0;p<cube_flat.size();p++){
				cube_flat[p]=cube[i][j][p];
			}
			myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);
			for (int p=0;p<residual_1D.size();p++){
				residual[j][i][p]=residual_1D[p];
			}
			if(std_map[i][j]>0.){
				f+=myfunc_spec(residual_1D)/pow(std_map[i][j],2.); //std_map est arrondie... 
			}
		}
	}
	int p, q;
	for(int i=0; i<M.n_gauss; i++){
		for(p=0; p<indice_y; p++){
			for(q=0; q<indice_x; q++){
				image_amp[p][q]=params[0+3*i][p][q];
				image_mu[p][q]=params[1+3*i][p][q];
				image_sig[p][q]=params[2+3*i][p][q];
			}
		}
	
		convolution_2D_mirror(M, image_amp, conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror(M, image_mu, conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror(M, image_sig, conv_sig, indice_y, indice_x,3);
	
		convolution_2D_mirror(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);

        int l,j;
	    #pragma omp parallel shared(M, params, residual, std_map, indice_x, indice_y,i,f,deriv, image_sig, b_params,conv_conv_amp, conv_conv_mu, conv_conv_sig)
	    {
        #pragma omp for private(l,j)
		for(l=0; l<indice_x; l++){
			for(j=0; j<indice_y; j++){
				f+= 0.5*M.lambda_amp*pow(conv_amp[j][l],2);
				f+= 0.5*M.lambda_mu*pow(conv_mu[j][l],2);
				f+= 0.5*M.lambda_sig*pow(conv_sig[j][l],2)+0.5*M.lambda_var_sig*pow(image_sig[j][l]-b_params[i],2);
	
				g[n_beta-M.n_gauss+i] += M.lambda_var_sig*(b_params[i]-image_sig[j][l]);
			
		int k;
		if(std_map[j][l]>0.){
//	    #pragma omp parallel shared(M, params, residual, std_map, indice_v,i,j,l)
//	    {
//        #pragma omp for private(k)
		for(k=0; k<indice_v; k++){
					deriv[l][j][0+3*i] += exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
	
					deriv[l][j][1+3*i] +=  params[3*i][j][l]*( double(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
	
					deriv[l][j][2+3*i] += params[3*i][j][l]*pow( double(k+1) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
					}
		}
//		}
				deriv[l][j][0+3*i] += M.lambda_amp*conv_conv_amp[j][l];
				deriv[l][j][1+3*i] += M.lambda_mu*conv_conv_mu[j][l];
				deriv[l][j][2+3*i] += M.lambda_sig*conv_conv_sig[j][l]+M.lambda_var_sig*(image_sig[j][l]-b_params[i]);
			}
		}
	}

	}
	ravel_3D_bis(deriv, g, 3*M.n_gauss, indice_y,indice_x);
}




//test optims OpenMP for f_g_cube
void algo_rohsa::f_g_cube_cuda_L(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig, double* cube_flattened) 
{

std::vector<std::vector<std::vector<double>>> dR_over_dB(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
//décommenter en bas

std::vector<std::vector<std::vector<double>>> g_3D(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));

std::vector<double> b_params(M.n_gauss,0.);

int i,k,j,l,p;

	int taille_params_flat[] = {indice_y, indice_x,3*M.n_gauss};
	int taille_deriv[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_residual[] = {indice_v, indice_y, indice_x};
	int taille_std_map_[] = {indice_y, indice_x};
	int taille_beta[] = {indice_x, indice_y, 3*M.n_gauss};
	int taille_beta_modif[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_cube[] = {indice_v, indice_y, indice_x};
	int taille_image_conv[] = {indice_y, indice_x};

	int product_residual = taille_residual[0]*taille_residual[1]*taille_residual[2];
	int product_params_flat = taille_params_flat[0]*taille_params_flat[1]*taille_params_flat[2];
	int product_deriv = taille_deriv[0]*taille_deriv[1]*taille_deriv[2];
	int product_std_map_ = taille_std_map_[0]*taille_std_map_[1];
	int product_cube = taille_cube[0]*taille_cube[1]*taille_cube[2]; 
	int product_beta = taille_beta[0]*taille_beta[1]*taille_beta[2];
	int product_beta_modif = taille_beta[0]*taille_beta[1]*taille_beta[2];
	int product_image_conv = taille_image_conv[0]*taille_image_conv[1];

	size_t size_deriv = product_deriv * sizeof(double);
	size_t size_res = product_residual * sizeof(double);
	size_t size_std = product_std_map_ * sizeof(double);
	size_t size_beta_modif = product_beta_modif * sizeof(double);
	size_t size_image_conv = product_image_conv * sizeof(double);

	double* deriv = (double*)malloc(size_deriv);
	double* residual = (double*)malloc(size_res);
	double* std_map_ = (double*)malloc(size_std);
	double* beta_modif = (double*)malloc(size_beta_modif);

	float* conv_amp = (float*)malloc(size_image_conv);
	float* conv_mu = (float*)malloc(size_image_conv);
	float* conv_sig = (float*)malloc(size_image_conv);
	float* conv_conv_amp = (float*)malloc(size_image_conv);
	float* conv_conv_mu = (float*)malloc(size_image_conv);
	float* conv_conv_sig = (float*)malloc(size_image_conv);
	float* image_amp = (float*)malloc(size_image_conv);
	float* image_mu = (float*)malloc(size_image_conv);
	float* image_sig = (float*)malloc(size_image_conv);


	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;
	double temps1_tableaux = omp_get_wtime();

	for(int i = 0; i<product_deriv; i++){
		deriv[i]=0.;
	}

	for(i=0; i<indice_y; i++){
		for(j=0; j<indice_x; j++){
			std_map_[i*indice_x+j]=std_map[i][j];
		}
	}


	for(int i = 0; i< n; i++){
		g[i]=0.;
	}
	f=0.;

//beta est de taille : x,y,3g
//params est de taille : 3g,y,x

	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

//exit(0);
	double temps_modification_beta1 = omp_get_wtime();

	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
			for(p=0; p<3*M.n_gauss; p++){
				beta_modif[p*indice_x*indice_y+ i*indice_x+j] = beta[j*indice_y*3*M.n_gauss+i*3*M.n_gauss+p];
			}
		}
	}

	double temps_modification_beta2 = omp_get_wtime();


double temps2_tableaux = omp_get_wtime();

double temps1_dF_dB = omp_get_wtime();



f =  compute_residual_and_f(beta_modif, taille_beta_modif, product_beta, cube_flattened, taille_cube, product_cube, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, indice_x, indice_y, indice_v, M.n_gauss);

double temps2_dF_dB = omp_get_wtime();
/*
	for(int i = 0; i<3; i++){
		printf("residual[%d]= %f \n",i, residual[i]);
	}

	for(int i = 0; i<10; i++){
		printf("beta_modif[%d]= %f \n",i, beta_modif[i]);
	}
	for(int i = 0; i<10; i++){
		printf("std_map_[%d]= %f \n",i, std_map_[i]);
	}
	for(int i = 0; i<10; i++){
		printf("deriv[%d]= %f \n",i, deriv[i]);
	}
*/
//std::cout<<"ENTRE BETA_MODIF ET LE GPU"	<<std::endl;

	double temps1_deriv = omp_get_wtime();


	gradient_L_2_beta(deriv, taille_deriv, product_deriv, beta_modif, taille_beta_modif, product_beta_modif, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, M.n_gauss);
/*
	for(int i = 0; i<10; i++){
		printf("deriv[%d]= %f \n",i, deriv[i]);
	}
*/
//printf("deriv[0] = %f", deriv[0]);
//exit(0);

    float Kernel_f[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};
    double Kernel_d[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};


//if(indice_x ==4){//indice_x != indice_y){
/*
	printf("après gradient_L_2_beta\n");
	printf("indice_x, indice_y = %d, %d\n", indice_x, indice_y);

	for(int i=0; i<30; i++){
		printf("deriv[%d] = %f \n",i,deriv[i]);
	}
//	std::cin.ignore();
//}

	printf("f = %f \n",f);
//}
*/

/*
	float h_IMAGE[1225];

	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
//			for(p=0; p<3*M.n_gauss; p++){
			h_IMAGE[i*indice_x+j] = (float)(beta_modif[0*indice_x*indice_y+ i*indice_x+j]);
//			}
		}
	}
	int N = 4;
	float IMAGE_TEST[16];
	double IMAGE_TEST_double[16];
	double IMAGE_TEST_double_cpu[16];

	for(j=0; j<N; j++){
		for(i=0; i<N; i++){
			IMAGE_TEST[j+N*i] = j+2*i;
			IMAGE_TEST_double_cpu[j+N*i] = j+2*i;
			IMAGE_TEST_double[j+N*i] = j+2*i;
		}
	}
	float RESULTAT_TEST_GPU_f[16];

	size_t size_RESULTAT_TEST_double = N*N * sizeof(double);

	double* RESULTAT_TEST_double_cpu = (double*)malloc(size_RESULTAT_TEST_double);
//	double* RESULTAT_TEST_GPU_double = (double*)malloc(size_RESULTAT_TEST_double);
	double RESULTAT_TEST_GPU_double[16];



	std::cout<< " "<<std::endl; 
	std::cout<< " "<<std::endl; 
	std::cout<< "conv2D_GPU"<<std::endl; 
	std::cout<< " "<<std::endl; 
	std::cout<< " "<<std::endl; 

	convolution_2D_mirror_flat(M, IMAGE_TEST_double_cpu, RESULTAT_TEST_double_cpu, N, N, 3);
	conv2D_GPU_all(IMAGE_TEST, Kernel_f, RESULTAT_TEST_GPU_f, N, N, 3, 0, 0);
	conv2D_GPU_all(beta_modif, IMAGE_TEST_double, Kernel_d, RESULTAT_TEST_GPU_double, N, N, 3, 0, 0);
//	conv2D_GPU_cpu(IMAGE_TEST, Kernel_f, RESULTAT_TEST_GPU, N, N, 0, 0);

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			std::cout<< "IMAGE_TEST["<<i*N+j << "] = "<< IMAGE_TEST[i*N+j]<<std::endl; 
		}
	}
	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			std::cout<< "IMAGE_TEST_double["<<i*N+j << "] = "<< IMAGE_TEST_double[i*N+j]<<std::endl; 
		}
	}

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			std::cout<< "RESULTAT_TEST_double_cpu["<<i*N+j << "] = "<< RESULTAT_TEST_double_cpu[i*N+j]<<std::endl; 
		}
	}

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			std::cout<< "RESULTAT_TEST_GPU_f["<<i*N+j << "] = "<< RESULTAT_TEST_GPU_f[i*N+j]<<std::endl; 
		}
	}

	for(i=0; i<N; i++){
		for(j=0; j<N; j++){
			std::cout<< "RESULTAT_TEST_GPU_double["<<i*N+j << "] = "<< RESULTAT_TEST_GPU_double[i*N+j]<<std::endl; 
		}
	}
exit(0);
*/


//	printf("deriv[0] = %f \n",deriv[0]);

	double temps2_deriv = omp_get_wtime();
	double temps1_conv = omp_get_wtime();

	for(int k=0; k<M.n_gauss; k++){
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){
				image_amp[p+indice_x*q]= (float)(beta[q*indice_y*3*M.n_gauss+ p*3*M.n_gauss+ (0+3*k)]);
				image_mu[p+indice_x*q]=(float)(beta[q*indice_y*3*M.n_gauss+ p*3*M.n_gauss+ (1+3*k)]);
				image_sig[p+indice_x*q]=(float)(beta[q*indice_y*3*M.n_gauss+ p*3*M.n_gauss+ (2+3*k)]);
			}
		}

if(false){//indice_x>=128 || indice_y>=128){
	conv2D_GPU(image_amp, Kernel_f, conv_amp, indice_x, indice_y, temps_transfert, temps_mirroirs);
	conv2D_GPU(image_mu, Kernel_f, conv_mu, indice_x, indice_y, temps_transfert, temps_mirroirs);
	conv2D_GPU(image_sig, Kernel_f, conv_sig, indice_x, indice_y, temps_transfert, temps_mirroirs);

	conv2D_GPU(conv_amp, Kernel_f, conv_conv_amp, indice_x, indice_y, temps_transfert, temps_mirroirs);
	conv2D_GPU(conv_mu, Kernel_f, conv_conv_mu, indice_x, indice_y, temps_transfert, temps_mirroirs);
	conv2D_GPU(conv_sig, Kernel_f, conv_conv_sig, indice_x, indice_y, temps_transfert, temps_mirroirs);

/*
printf("-> image_amp[0] = %f \n",image_amp[0]);
printf("-> image_amp[1] = %f \n",image_amp[1]);

printf("image_amp[0] = %f \n",image_amp[0]);
printf("image_amp[1] = %f \n",image_amp[1]);
printf("conv_amp[0] = %f \n",conv_amp[0]);
printf("conv_amp[1] = %f \n",conv_amp[1]);
*/
/*
    printf("\n IMAGE \n");

	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			printf("image_amp[%d][%d] = %f \n",i,j,image_amp[i+indice_x*j]);
		}
	}
	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			printf("image_mu[%d][%d] = %f \n",i,j,image_mu[i+indice_x*j]);
		}
	}
	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			printf("image_sig[%d][%d] = %f \n",i,j,image_sig[i+indice_x*j]);
		}
	}

	


    printf("\n CONV \n");

	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			printf("conv_amp[%d][%d] = %f \n",i,j,conv_amp[i+indice_x*j]);
		}
	}
	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			printf("conv_mu[%d][%d] = %f \n",i,j,conv_mu[i+indice_x*j]);
		}
	}
	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			printf("conv_sig[%d][%d] = %f \n",i,j,conv_sig[i+indice_x*j]);
		}
	}

    printf("\n CONV_CONV \n");

	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			printf("conv_conv_amp[%d][%d] = %f \n",i,j,conv_conv_amp[i+indice_x*j]);
		}
	}
	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			printf("conv_conv_mu[%d][%d] = %f \n",i,j,conv_conv_mu[i+indice_x*j]);
		}
	}
	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			printf("conv_conv_sig[%d][%d] = %f \n",i,j,conv_conv_sig[i+indice_x*j]);
		}
	}
*/
//	std::cin.ignore();

//exit(0);
} else{
	convolution_2D_mirror_flat(M, image_amp, conv_amp, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
	convolution_2D_mirror_flat(M, image_mu, conv_mu, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
	convolution_2D_mirror_flat(M, image_sig, conv_sig, indice_y, indice_x,3, temps_transfert, temps_mirroirs);

	convolution_2D_mirror_flat(M, conv_amp, conv_conv_amp, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
	convolution_2D_mirror_flat(M, conv_mu, conv_conv_mu, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
	convolution_2D_mirror_flat(M, conv_sig, conv_conv_sig, indice_y, indice_x,3, temps_transfert, temps_mirroirs);

//if(k <= 1 && indice_x!=indice_y)
//{
//if(indice_x ==4){//indice_x != indice_y){

/*
    printf("\n IMAGE cuda_L\n");
	for(int i=0; i<indice_x; i++){
		for(int j=0; j<indice_y; j++){
			printf("image_amp[%d][%d] = %f \n",i,j,image_amp[i+indice_x*j]);
		}
	}
    printf("\n CONV cuda_L\n");
	for(int i=0; i<indice_x; i++){
		for(int j=0; j<indice_y; j++){
			printf("conv_amp[%d][%d] = %f \n",i,j,conv_amp[i+indice_x*j]);
		}
	}
*/

//}
//	std::cin.ignore();

//	exit(0);
//}
}

	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			f+= 0.5*M.lambda_amp*pow((double)(conv_amp[i+indice_x*j]),2);
			f+= 0.5*M.lambda_mu*pow((double)(conv_mu[i+indice_x*j]),2);
			f+= 0.5*M.lambda_sig*pow((double)(conv_sig[i+indice_x*j]),2) + 0.5*M.lambda_var_sig*pow((double)(image_sig[i+indice_x*j])-b_params[k],2);

			deriv[(0+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_amp*(double)(conv_conv_amp[i+indice_x*j]);
			deriv[(1+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_mu*(double)(conv_conv_mu[i+indice_x*j]);

			deriv[(2+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_sig*(double)(conv_conv_sig[i+indice_x*j])+M.lambda_var_sig*((double)(image_sig[i+indice_x*j])-b_params[k]);
/*
			dR_over_dB[0+3*k][i][j] = M.lambda_amp*(double)(conv_conv_amp[i+indice_x*j]);
			dR_over_dB[1+3*k][i][j] = M.lambda_mu*(double)(conv_conv_mu[i+indice_x*j]);
			dR_over_dB[2+3*k][i][j] = M.lambda_sig*(double)(conv_conv_sig[i+indice_x*j])+M.lambda_var_sig*((double)(image_sig[i+indice_x*j])-b_params[k]);
*/
			g[n_beta-M.n_gauss+k] += M.lambda_var_sig*(b_params[k]-(double)(image_sig[i+indice_x*j]));
		}
	}


//if(indice_x != indice_y || (indice_x == 128 && indice_x == 128)){
//}
	//£
//	printf("deriv[0] = %f \n",deriv[0] );
//	exit(0);
	}


	for(int l=0; l<3*M.n_gauss; l++){
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				g_3D[l][i][j] = deriv[l*indice_y*indice_x+i*indice_x+j];// + dR_over_dB[l][i][j];
			}
		}
	}
	ravel_3D(g_3D, g, 3*M.n_gauss, indice_y, indice_x);

/*
//if(indice_x ==4){//indice_x != indice_y){
	printf("après conv \n");
	for(int i = 0; i<20; i++){
		printf("b_params[%d] = %f\n", i, b_params[i]);
	}

	for(int i=0; i<indice_x*indice_y*3*M.n_gauss; i++){
		printf("g[%d] = %f \n",i,g[i]);
	}
	printf("f = %f \n",f);
	std::cin.ignore();
//}
*/

/*
	for(int i = 0; i<product_deriv; i++){
		g[i] = deriv[i];
	}
*/
//	g = deriv;//À TESTER

/*
	printf("g[0] = %f \n",g[0]);
	printf("g[1] = %f \n",g[1]);
	printf("g[2] = %f \n",g[2]);
	std::cin.ignore();
*/

	double temps2_conv = omp_get_wtime();

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;

	temps_modification_beta += temps_modification_beta2 - temps_modification_beta1;

	temps_f_g_cube += temps2_dF_dB - temps1_dF_dB;

	free(beta_modif);

	free(deriv);
	free(residual);
	free(std_map_);

	free(conv_amp);
	free(conv_conv_amp);
	free(conv_mu);
	free(conv_conv_mu);
	free(conv_sig);
	free(conv_conv_sig);
	free(image_sig);
	free(image_mu);
	free(image_amp);
}


//test optims OpenMP for f_g_cube
void algo_rohsa::f_g_cube_cuda_L_deux_tiers(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig, double* cube_flattened) 
{

std::vector<std::vector<std::vector<double>>> dR_over_dB(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));

//décommenter en bas

std::vector<std::vector<std::vector<double>>> g_3D(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<std::vector<std::vector<double>>> params(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<double> b_params(M.n_gauss,0.);

std::vector<std::vector<double>> conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_sig(indice_y,std::vector<double>(indice_x, 0.));

int i,k,j,l,p;

	int taille_params_flat[] = {indice_y, indice_x,3*M.n_gauss};
	int taille_deriv[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_residual[] = {indice_x, indice_y, indice_v};
	int taille_std_map_[] = {indice_y, indice_x};
	int taille_beta[] = {indice_x, indice_y, 3*M.n_gauss};
	int taille_cube[] = {indice_x, indice_y, indice_v};

	int product_residual = taille_residual[0]*taille_residual[1]*taille_residual[2];
	int product_params_flat = taille_params_flat[0]*taille_params_flat[1]*taille_params_flat[2];
	int product_deriv = taille_deriv[0]*taille_deriv[1]*taille_deriv[2];
	int product_std_map_ = taille_std_map_[0]*taille_std_map_[1];
	int product_cube = taille_cube[0]*taille_cube[1]*taille_cube[2]; 
	int product_beta = taille_beta[0]*taille_beta[1]*taille_beta[2];

	size_t size_deriv = product_deriv * sizeof(double);
	size_t size_res = product_residual * sizeof(double);
	size_t size_std = product_std_map_ * sizeof(double);
	size_t size_params = product_params_flat*sizeof(double);

	double* params_flat = (double*)malloc(size_params);
	double* deriv = (double*)malloc(size_deriv);
	double* residual = (double*)malloc(size_res);
	double* std_map_ = (double*)malloc(size_std);

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;
	double temps1_tableaux = omp_get_wtime();

	for(j=0; j<indice_y; j++)
	{
		for(i=0; i<indice_x; i++)
		{
			for(k=0; k<3*M.n_gauss; k++)
			{
				params_flat[j*3*M.n_gauss*indice_x+i*3*M.n_gauss+k] = beta[i*indice_y*3*M.n_gauss+j*3*M.n_gauss+k];
			}
		}
	}


	for(int i = 0; i<product_deriv; i++){
		deriv[i]=0.;
	}

	for(i=0; i<indice_y; i++){
		for(j=0; j<indice_x; j++){
			std_map_[i*indice_x+j]=std_map[i][j];
		}
	}


	for(int i = 0; i< n; i++){
		g[i]=0.;
	}
	f=0.;

//beta est de taille : x,y,3g
//params est de taille : 3g,y,x

	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

//exit(0);
double temps2_tableaux = omp_get_wtime();

double temps1_dF_dB = omp_get_wtime();



f = compute_residual_and_f(beta, taille_beta, product_beta, cube_flattened, taille_cube, product_cube, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, indice_x, indice_y, indice_v, M.n_gauss);



double temps2_dF_dB = omp_get_wtime();



	double temps1_deriv = omp_get_wtime();

//AJOUT
//-----------------------------------------------------------------------------------
//params    3g y  x
//params_T  y  x 3g
//dF_over_dB_bis[0+3*i][k][j][l] += exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

	gradient_L_2(deriv, taille_deriv, product_deriv, params_flat, taille_params_flat, product_params_flat, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, M.n_gauss);

	double temps2_deriv = omp_get_wtime();

	double temps1_conv = omp_get_wtime();

	for(int k=0; k<M.n_gauss; k++){
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){

				image_amp[p][q]=params_flat[p*indice_x*3*M.n_gauss+q*3*M.n_gauss+ (0+3*k)];
				image_mu[p][q]=params_flat[p*indice_x*3*M.n_gauss+q*3*M.n_gauss+ (1+3*k)];
				image_sig[p][q]=params_flat[p*indice_x*3*M.n_gauss+q*3*M.n_gauss+ (2+3*k)];
			}
		}

	convolution_2D_mirror(M, image_amp, conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, image_mu, conv_mu, indice_y, indice_x,3);
	convolution_2D_mirror(M, image_sig, conv_sig, indice_y, indice_x,3);

	convolution_2D_mirror(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);


	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			f+= 0.5*M.lambda_amp*pow(conv_amp[i][j],2);
			f+= 0.5*M.lambda_mu*pow(conv_mu[i][j],2);
			f+= 0.5*M.lambda_sig*pow(conv_sig[i][j],2) + 0.5*M.lambda_var_sig*pow(image_sig[i][j]-b_params[k],2);

			dR_over_dB[0+3*k][i][j] = M.lambda_amp*conv_conv_amp[i][j];
			dR_over_dB[1+3*k][i][j] = M.lambda_mu*conv_conv_mu[i][j];
			dR_over_dB[2+3*k][i][j] = M.lambda_sig*conv_conv_sig[i][j]+M.lambda_var_sig*(image_sig[i][j]-b_params[k]);

			g[n_beta-M.n_gauss+k] += M.lambda_var_sig*(b_params[k]-image_sig[i][j]);
		}
	}
	}

	for(int l=0; l<3*M.n_gauss; l++){
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				g_3D[l][i][j] = deriv[l*indice_y*indice_x+i*indice_x+j] + dR_over_dB[l][i][j];
			}
		}
	}
	ravel_3D(g_3D, g, 3*M.n_gauss, indice_y, indice_x);

	double temps2_conv = omp_get_wtime();

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;

	temps_f_g_cube += temps2_dF_dB - temps1_dF_dB;

	free(deriv);
	free(residual);
	free(std_map_);
	free(params_flat);
}


void algo_rohsa::f_g_cube_cuda_L_2(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig) 
{

std::vector<std::vector<std::vector<double>>> dR_over_dB(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));

//décommenter en bas

std::vector<std::vector<std::vector<std::vector<double>>>> dF_over_dB_bis(3*M.n_gauss,std::vector<std::vector<std::vector<double>>>( indice_v,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.))));
std::vector<std::vector<std::vector<double>>> deriv_bis(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<std::vector<std::vector<double>>> g_3D(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<std::vector<std::vector<double>>> residual_bis(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_v,0.)));
std::vector<std::vector<std::vector<double>>> params(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<double> b_params(M.n_gauss,0.);

//TEST VALIDITÉ GPU
/*
std::vector<std::vector<std::vector<std::vector<double>>>> dF_over_dB_bis(indice_v,std::vector<std::vector<std::vector<double>>>( indice_y,std::vector<std::vector<double>>(indice_x, std::vector<double>(3*M.n_gauss,0.))));
std::vector<std::vector<std::vector<double>>> deriv_bis(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
*/
std::vector<std::vector<std::vector<double>>> params_T(indice_y,std::vector<std::vector<double>>(indice_x, std::vector<double>(3*M.n_gauss,0.)));

std::vector<std::vector<double>> conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_sig(indice_y,std::vector<double>(indice_x, 0.));

int i,k,j,l,p;

int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;
double temps1_tableaux = omp_get_wtime();
for(int i = 0; i< n; i++){
	g[i]=0.;
}
f=0.;

unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);


for(int i = 0; i<M.n_gauss; i++){
	b_params[i]=beta[n_beta-M.n_gauss+i];
}

for(int j=0; j<indice_x; j++){
	for(int i=0; i<indice_y; i++){
		std::vector<double> residual_1D(indice_v,0.);
		std::vector<double> params_flat(params.size(),0.);
		std::vector<double> cube_flat(cube[0][0].size(),0.);
		for (int p=0;p<params_flat.size();p++){
			params_flat[p]=params[p][i][j];
		}
		for (int p=0;p<cube_flat.size();p++){
			cube_flat[p]=cube[j][i][p];
		}
		myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);
		for (int p=0;p<residual_1D.size();p++){
			residual_bis[j][i][p]=residual_1D[p];
		}
		if(std_map[i][j]>0.){
			f += myfunc_spec(residual_1D)*1/pow(std_map[i][j],2.); //std_map est arrondie... 
		}
	}
}



double temps2_tableaux = omp_get_wtime();

double temps1_dF_dB = omp_get_wtime();





double temps2_dF_dB = omp_get_wtime();

double temps1_deriv = omp_get_wtime();

//AJOUT
//-----------------------------------------------------------------------------------
//params    3g y  x
//params_T  y  x 3g
//dF_over_dB_bis[0+3*i][k][j][l] += exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

	for(k=0; k<3*M.n_gauss; k++)
	{
		for(j=0; j<indice_y; j++)
		{
			for(i=0; i<indice_x; i++)
			{
				params_T[j][i][k] = params[k][j][i];
			}
		}
	}

	int taille_params_flat[] = {indice_y, indice_x,3*M.n_gauss};
	int taille_deriv[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_residual[] = {indice_x, indice_y, indice_v};
	int taille_std_map_[] = {indice_y, indice_x};

	int product_residual = taille_residual[0]*taille_residual[1]*taille_residual[2];
	int product_params_flat = taille_params_flat[0]*taille_params_flat[1]*taille_params_flat[2];
	int product_deriv = taille_deriv[0]*taille_deriv[1]*taille_deriv[2];
	int product_std_map_ = taille_std_map_[0]*taille_std_map_[1];
		
	size_t size_deriv = product_deriv * sizeof(double);
	size_t size_res = product_residual * sizeof(double);
	size_t size_std = product_std_map_ * sizeof(double);
	size_t size_params = product_params_flat*sizeof(double);

	double* params_flat = (double*)malloc(size_params);
	double* deriv = (double*)malloc(size_deriv);
	double* residual = (double*)malloc(size_res);
	double* std_map_ = (double*)malloc(size_std);

	for(int i = 0; i<product_deriv; i++){
		deriv[i]=0.;
	}
	for(i=0; i<indice_y; i++){
		for(j=0; j<indice_x; j++){
			std_map_[i*indice_x+j]=std_map[i][j];
		}
	}

	for(int j=0; j<indice_x; j++){
		for(int i=0; i<indice_y; i++){
			for (int p=0;p<indice_v;p++){
				residual[j*indice_y*indice_v+i*indice_v+p]=residual_bis[j][i][p];
			}
		}
	}


	for(i=0; i<indice_y; i++){
		for(j=0; j<indice_x; j++){
			for (p=0;p<3*M.n_gauss;p++){
				params_flat[i*3*M.n_gauss*indice_x+j*3*M.n_gauss+p]=params_T[i][j][p];
			}
		}
	}

	gradient_L_2(deriv, taille_deriv, product_deriv, params_flat, taille_params_flat, product_params_flat, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, M.n_gauss);

/*
printf("beta[0] = %f \n",beta[0]);
printf("beta[1] = %f \n",beta[1]);
printf("beta[2] = %f \n",beta[2]);
printf("beta[3] = %f \n",beta[3]);
*/

	double temps1_conv = omp_get_wtime();

	for(int k=0; k<M.n_gauss; k++){
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){
				image_amp[p][q]=params_flat[p*indice_x*3*M.n_gauss+q*3*M.n_gauss+ (0+3*k)];
				image_mu[p][q]=params_flat[p*indice_x*3*M.n_gauss+q*3*M.n_gauss+ (1+3*k)];
				image_sig[p][q]=params_flat[p*indice_x*3*M.n_gauss+q*3*M.n_gauss+ (2+3*k)];
			}
		}

	convolution_2D_mirror(M, image_amp, conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, image_mu, conv_mu, indice_y, indice_x,3);
	convolution_2D_mirror(M, image_sig, conv_sig, indice_y, indice_x,3);

/*
printf("-> image_amp[0] = %f \n",image_amp[0][0]);
printf("-> image_amp[1] = %f \n",image_amp[0][1]);

printf("image_amp[0] = %f \n",image_amp[0][0]);
printf("image_amp[1] = %f \n",image_amp[0][1]);
	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			printf("conv_amp[i][j] = %f \n",conv_amp[i][j]);
		}
	}

	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			printf("conv_mu[i][j] = %f \n",conv_mu[i][j]);
		}
	}
	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			printf("conv_sig[i][j] = %f \n",conv_sig[i][j]);
		}
	}
//	exit(0);
std::cin.ignore();
*/
	convolution_2D_mirror(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);


	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			f+= 0.5*M.lambda_amp*pow(conv_amp[i][j],2);
			f+= 0.5*M.lambda_mu*pow(conv_mu[i][j],2);
			f+= 0.5*M.lambda_sig*pow(conv_sig[i][j],2) + 0.5*M.lambda_var_sig*pow(image_sig[i][j]-b_params[k],2);

			dR_over_dB[0+3*k][i][j] = M.lambda_amp*conv_conv_amp[i][j];
			dR_over_dB[1+3*k][i][j] = M.lambda_mu*conv_conv_mu[i][j];
			dR_over_dB[2+3*k][i][j] = M.lambda_sig*conv_conv_sig[i][j]+M.lambda_var_sig*(image_sig[i][j]-b_params[k]);

			g[n_beta-M.n_gauss+k] += M.lambda_var_sig*(b_params[k]-image_sig[i][j]);
		}
	}
	}

	for(int l=0; l<3*M.n_gauss; l++){
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				g_3D[l][i][j] = deriv[l*indice_y*indice_x+i*indice_x+j] + dR_over_dB[l][i][j];
			}
		}
	}
	ravel_3D(g_3D, g, 3*M.n_gauss, indice_y, indice_x);
	/*
	printf("g[0] = %f \n",g[0]);
	printf("g[1] = %f \n",g[1]);
	printf("g[2] = %f \n",g[2]);
	std::cin.ignore();
	*/

	double temps2_conv = omp_get_wtime();

	temps_conv+= temps2_conv - temps1_conv;
//	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;

	temps_f_g_cube += temps2_dF_dB - temps1_dF_dB;

	free(deriv);
	free(residual);
	free(std_map_);
	free(params_flat);
}


void algo_rohsa::f_g_cube_cuda_L_2_2Bverified(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig) 
{

std::vector<std::vector<std::vector<double>>> dR_over_dB(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));

//décommenter en bas

std::vector<std::vector<std::vector<std::vector<double>>>> dF_over_dB_bis(3*M.n_gauss,std::vector<std::vector<std::vector<double>>>( indice_v,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.))));
std::vector<std::vector<std::vector<double>>> deriv_bis(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<std::vector<std::vector<double>>> g_3D(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<std::vector<std::vector<double>>> residual_bis(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_v,0.)));
std::vector<std::vector<std::vector<double>>> params(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<double> b_params(M.n_gauss,0.);

//TEST VALIDITÉ GPU
/*
std::vector<std::vector<std::vector<std::vector<double>>>> dF_over_dB_bis(indice_v,std::vector<std::vector<std::vector<double>>>( indice_y,std::vector<std::vector<double>>(indice_x, std::vector<double>(3*M.n_gauss,0.))));
std::vector<std::vector<std::vector<double>>> deriv_bis(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
*/
std::vector<std::vector<std::vector<double>>> params_T(indice_y,std::vector<std::vector<double>>(indice_x, std::vector<double>(3*M.n_gauss,0.)));

std::vector<std::vector<double>> conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_sig(indice_y,std::vector<double>(indice_x, 0.));

int i,k,j,l,p;

int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;
double temps1_tableaux = omp_get_wtime();
for(int i = 0; i< n; i++){
	g[i]=0.;
}
f=0.;

unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);


for(int i = 0; i<M.n_gauss; i++){
	b_params[i]=beta[n_beta-M.n_gauss+i];
}

for(int j=0; j<indice_x; j++){
	for(int i=0; i<indice_y; i++){
		std::vector<double> residual_1D(indice_v,0.);
		std::vector<double> params_flat(params.size(),0.);
		std::vector<double> cube_flat(cube[0][0].size(),0.);
		for (int p=0;p<params_flat.size();p++){
			params_flat[p]=params[p][i][j];
		}
		for (int p=0;p<cube_flat.size();p++){
			cube_flat[p]=cube[j][i][p];
		}
		myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);
		for (int p=0;p<residual_1D.size();p++){
			residual_bis[j][i][p]=residual_1D[p];
		}
		if(std_map[i][j]>0.){
			f += myfunc_spec(residual_1D)*1/pow(std_map[i][j],2.); //std_map est arrondie... 
		}
	}
}

double temps2_tableaux = omp_get_wtime();

double temps1_dF_dB = omp_get_wtime();



/*
//-----------------------------------------------------------------------------

#pragma omp parallel private(i,k) shared(dF_over_dB_bis ,params ,M ,indice_v ,indice_y ,indice_x) //num_threads(1) //
{
#pragma omp for
for(int i=0; i<M.n_gauss; i++){
        for(k=0; k<indice_v; k++){
                for(int j=0; j<indice_y; j++){
                        for(int l=0; l<indice_x; l++){

				dF_over_dB_bis[0+3*i][k][j][l] += exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

				dF_over_dB_bis[1+3*i][k][j][l] +=  params[3*i][j][l]*( double(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * 
									exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

				dF_over_dB_bis[2+3*i][k][j][l] += params[3*i][j][l]*pow( double(k+1) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * 
									exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

                        }
                }
        }
}
}

double temps2_dF_dB = omp_get_wtime();

double temps1_deriv = omp_get_wtime();
#pragma omp parallel private(k,j) shared(deriv_bis, dF_over_dB_bis, M, indice_v, indice_y, indice_x, std_map, residual_bis)
{
#pragma omp for
for(k=0; k<indice_v; k++){
	for(j=0; j<indice_x; j++){
		for(int i=0; i<indice_y; i++){
			for(int l=0; l<3*M.n_gauss; l++){
				if(std_map[i][j]>0.){
					deriv_bis[l][i][j]+=  dF_over_dB_bis[l][k][i][j]*residual_bis[j][i][k]/pow(std_map[i][j],2);
				}
			}
		}
	}
}
}

double temps2_deriv = omp_get_wtime();
*/
//-----------------------------------------------------------------------------


double temps2_dF_dB = omp_get_wtime();

double temps1_deriv = omp_get_wtime();
//	int i,j,l,k;
int compteur_cpu = 0;
//#pragma omp parallel private(i,l,j) shared(params, deriv_bis, M, indice_v, indice_y, indice_x, std_map, residual_bis)
//{
	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
			for(l=0; l<M.n_gauss; l++){
				if(std_map[i][j]>0.){
				double temp1=0.;
				double temp2=0.;
				double temp3=0.;
				double res_std=0.;
					for(int k_=0; k_<indice_v; k_++){
					res_std = residual_bis[j][i][k_]/pow(std_map[i][j],2);
					temp1 += exp(-pow( double(k_+1)-params[1+3*l][i][j],2.)/(2*pow(params[2+3*l][i][j],2.)) )*res_std;
					temp2 +=  params[3*l][i][j]*( double(k_+1) - params[1+3*l][i][j])/pow(params[2+3*l][i][j],2.) * 
							exp(-pow( double(k_+1)-params[1+3*l][i][j],2.)/(2*pow(params[2+3*l][i][j],2.)) )*res_std;
					temp3 += params[3*l][i][j]*pow( double(k_+1) - params[1+3*l][i][j], 2.)/(pow(params[2+3*l][i][j],3.)) * 
							exp(-pow( double(k_+1)-params[1+3*l][i][j],2.)/(2*pow(params[2+3*l][i][j],2.)) )*res_std;
/*
dF_over_dB_bis[0+3*l][k][i][j] += exp(-pow( double(k+1)-params[1+3*l][i][j],2.)/(2*pow(params[2+3*l][i][j],2.)) );
dF_over_dB_bis[1+3*l][k][i][j] +=  params[3*l][i][j]*( double(k+1) - params[1+3*l][i][j])/pow(params[2+3*l][i][j],2.) * 
						exp(-pow( double(k+1)-params[1+3*l][i][j],2.)/(2*pow(params[2+3*l][i][j],2.)) );
dF_over_dB_bis[2+3*l][k][i][j] += params[3*l][i][j]*pow( double(k+1) - params[1+3*l][i][j], 2.)/(pow(params[2+3*l][i][j],3.)) * 
						exp(-pow( double(k+1)-params[1+3*l][i][j],2.)/(2*pow(params[2+3*l][i][j],2.)) );
*/
					}



//				printf("temp1 = %f \n", temp1);
//				printf("temp1 = %d , temp2 = %d , temp3 = %d\n", temp1, temp2, temp3);

				deriv_bis[3*l+0][i][j]+=  temp1;
				deriv_bis[3*l+1][i][j]+=  temp2;
				deriv_bis[3*l+2][i][j]+=  temp3;
				}
				printf("j = %d , i = %d , l = %d\n",j,i,l);
//				compteur_cpu++;
			}
		}
	}
//}

std::cout<<"compteur_cpu = "<<compteur_cpu<<std::endl;

//AJOUT
//-----------------------------------------------------------------------------------
//params    3g y  x
//params_T  y  x 3g
//dF_over_dB_bis[0+3*i][k][j][l] += exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );


	for(k=0; k<3*M.n_gauss; k++)
	{
		for(j=0; j<indice_y; j++)
		{
			for(i=0; i<indice_x; i++)
			{
				params_T[j][i][k] = params[k][j][i];
			}
		}
	}

	int taille_params_flat[] = {indice_y, indice_x,3*M.n_gauss};
	int taille_deriv[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_residual[] = {indice_x, indice_y, indice_v};
	int taille_std_map_[] = {indice_y, indice_x};

	int product_residual = taille_residual[0]*taille_residual[1]*taille_residual[2];
	int product_params_flat = taille_params_flat[0]*taille_params_flat[1]*taille_params_flat[2];
	int product_deriv = taille_deriv[0]*taille_deriv[1]*taille_deriv[2];
	int product_std_map_ = taille_std_map_[0]*taille_std_map_[1];
		
	size_t size_deriv = product_deriv * sizeof(double);
	size_t size_res = product_residual * sizeof(double);
	size_t size_std = product_std_map_ * sizeof(double);
	size_t size_params = product_params_flat*sizeof(double);

	double* params_flat = (double*)malloc(size_params);
	double* deriv = (double*)malloc(size_deriv);
	double* residual = (double*)malloc(size_res);
	double* std_map_ = (double*)malloc(size_std);

	for(int i = 0; i<product_deriv; i++){
		deriv[i]=0.;
	}
	for(i=0; i<indice_y; i++){
		for(j=0; j<indice_x; j++){
			std_map_[i*indice_x+j]=std_map[i][j];
		}
	}

	for(int j=0; j<indice_x; j++){
		for(int i=0; i<indice_y; i++){
			for (int p=0;p<indice_v;p++){
				residual[j*indice_y*indice_v+i*indice_v+p]=residual_bis[j][i][p];
			}
		}
	}


	for(i=0; i<indice_y; i++){
		for(j=0; j<indice_x; j++){
			for (p=0;p<3*M.n_gauss;p++){
				params_flat[i*3*M.n_gauss*indice_x+j*3*M.n_gauss+p]=params_T[i][j][p];
			}
		}
	}
	gradient_L_2(deriv, taille_deriv, product_deriv, params_flat, taille_params_flat, product_params_flat, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, M.n_gauss);
	int vieux_compteur = 0;
	for(int l=0; l<3*M.n_gauss; l++){
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
//				if (abs(deriv_bis[l][i][j]-deriv[l*indice_x*indice_y+i*indice_x+j])!=0)
//				{

//					std::cout<< "deriv_bis[l][i][j]                      = "<<deriv_bis[l][i][j]<<std::endl;
//					std::cout<< "deriv[l*indice_x*indice_y+i*indice_x+j] = "<<deriv[l*indice_x*indice_y+i*indice_x+j]<<std::endl;

//					vieux_compteur++;
//				}
			}
		}
	}
	std::cout<<"--> vieux_compteur =  "<<vieux_compteur<<std::endl;
	free(deriv);
	free(residual);
	free(std_map_);
	free(params_flat);

//	exit(0);
//-----------------------------------------------------------------------------------


double temps1_conv = omp_get_wtime();

for(int k=0; k<M.n_gauss; k++){
	for(int p=0; p<indice_y; p++){
		for(int q=0; q<indice_x; q++){
			image_amp[p][q]=params[0+3*k][p][q];
			image_mu[p][q]=params[1+3*k][p][q];
			image_sig[p][q]=params[2+3*k][p][q];
		}
	}

	convolution_2D_mirror(M, image_amp, conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, image_mu, conv_mu, indice_y, indice_x,3);
	convolution_2D_mirror(M, image_sig, conv_sig, indice_y, indice_x,3);

	convolution_2D_mirror(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);


	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			f+= 0.5*M.lambda_amp*pow(conv_amp[i][j],2);
			f+= 0.5*M.lambda_mu*pow(conv_mu[i][j],2);
			f+= 0.5*M.lambda_sig*pow(conv_sig[i][j],2) + 0.5*M.lambda_var_sig*pow(image_sig[i][j]-b_params[k],2);

			dR_over_dB[0+3*k][i][j] = M.lambda_amp*conv_conv_amp[i][j];
			dR_over_dB[1+3*k][i][j] = M.lambda_mu*conv_conv_mu[i][j];
			dR_over_dB[2+3*k][i][j] = M.lambda_sig*conv_conv_sig[i][j]+M.lambda_var_sig*(image_sig[i][j]-b_params[k]);

			g[n_beta-M.n_gauss+k] += M.lambda_var_sig*(b_params[k]-image_sig[i][j]);
		}
	}
	}

	for(int l=0; l<3*M.n_gauss; l++){
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				g_3D[l][i][j] = deriv_bis[l][i][j] + dR_over_dB[l][i][j];
			}
		}
	}
	ravel_3D(g_3D, g, 3*M.n_gauss, indice_y, indice_x);

	double temps2_conv = omp_get_wtime();

	temps_conv+= temps2_conv - temps1_conv;
//	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;

	temps_f_g_cube += temps2_dF_dB - temps1_dF_dB;
}


//test optims OpenMP for f_g_cube
void algo_rohsa::f_g_cube_omp(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig) 
{

//std::cout<<"Début f_g_cube_test"<<std::endl;

std::vector<std::vector<std::vector<double>>> dR_over_dB(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
//°
//décommenter en bas
std::vector<std::vector<std::vector<std::vector<double>>>> dF_over_dB_bis(indice_v,std::vector<std::vector<std::vector<double>>>( indice_y,std::vector<std::vector<double>>(indice_x, std::vector<double>(3*M.n_gauss,0.))));
std::vector<std::vector<std::vector<double>>> deriv(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
//std::vector<std::vector<std::vector<double>>> deriv_bis(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<std::vector<std::vector<double>>> g_3D(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
//dF_over_dB[v][y][x][M.n__gauss]
//deriv[gauss][y][x]
std::vector<std::vector<std::vector<double>>> residual(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_v,0.)));

std::vector<std::vector<std::vector<double>>> params(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<std::vector<std::vector<double>>> params_T(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(3*M.n_gauss,0.)));
std::vector<double> b_params(M.n_gauss,0.);

std::vector<std::vector<double>> conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_sig(indice_y,std::vector<double>(indice_x, 0.));

int n_beta = (3*M.n_gauss*indice_x*indice_y);//+M.n_gauss;

int i,k,j,l,p;

for(int i = 0; i< n; i++){
	g[i]=0.;
}
f=0.;

double temps1_ravel = omp_get_wtime();

//extracting one params_T for the gradient and another for calculations below
// WARNING
unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

unravel_3D_T(beta, params_T,  3*M.n_gauss, indice_x,indice_y);


int N_x = indice_x;
int N_y = indice_y;
int N_z = 3*M.n_gauss;

/*
std::cout<<"1 : "<<params_T[indice_y-1][indice_x-1][3*M.n_gauss-1]<<std::endl;

std::cout<<"2 : "<<params[3*M.n_gauss-1][0][1]<<std::endl;
std::cout<<"1 : "<<params_T[0][1][3*M.n_gauss-1]<<std::endl;
*/
//exit(0);
double temps2_ravel = omp_get_wtime();
temps_ravel+=temps2_ravel-temps1_ravel;

/*for(int i = 0; i<M.n_gauss; i++){
	b_params[i]=beta[n_beta-M.n_gauss+i];
}*/
//cout.precision(dbl::max_digits10);


//#pragma omp parallel private(j,i) shared(cube,params,std_map,residual,indice_x,indice_y,indice_v,M,f)
//{
//#pragma omp for
double temps1_tableaux = omp_get_wtime();
for(j=0; j<indice_x; j++){
	for(i=0; i<indice_y; i++){
		std::vector<double> residual_1D(indice_v,0.);
		std::vector<double> params_flat(params.size(),0.);
		std::vector<double> cube_flat(cube[0][0].size(),0.);
		for (p=0;p<params_flat.size();p++){
			params_flat[p]=params[p][i][j];
		}
		for (p=0;p<cube_flat.size();p++){
			cube_flat[p]=cube[i][j][p];
		}
		myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);
		for (p=0;p<residual_1D.size();p++){
			residual[j][i][p]=residual_1D[p];
		}
		if(std_map[i][j]>0.){
			f+=myfunc_spec(residual_1D)/pow(std_map[i][j],2.); //std_map est arrondie... 
		}
	}

//}
}

double temps2_tableaux = omp_get_wtime();

double temps1_dF_dB = omp_get_wtime();


// µµ

/*
for(i=0; i<indice_y; i++){
	for(j=0; j<indice_x; j++){
		for (p=0;p<3*M.n_gauss;p++){
			std::cout<<params_T[i][j][p]<<std::endl;
			std::cout<<params_flat[p+3*M.n_gauss*j+3*M.n_gauss*indice_x*i]<<std::endl;
			std::cout<<beta[p+3*M.n_gauss*i+3*M.n_gauss*indice_x*j]<<std::endl;
			std::cout<<"  "<<std::endl;
		}
	}
}
*/

/*
for(int l=0; l<indice_y; l++){
	for(int i=0; i<indice_x; i++){
		for(int j=0; j<3*M.n_gauss; j++){
	std::cout<<"params_T   ["<<l<<"]["<<i<<"]["<< j <<"] ="<< params_T[l][i][j]<<std::endl;
	std::cout<<"params_flat["<<l<<"]["<<i<<"]["<< j <<"] ="<< params_flat[l*t3*t2+i*t3+j]<<std::endl;
	std::cout<<"  "<<std::endl;
		}
	}
}
*/

//gradient.cu gpu
////gradient(dF_over_dB, taille_dF_over_dB, product_taille_dF_over_dB, beta, taille_params_flat, product_taille_params_flat, M.n_gauss);

/*
   for(int p; p<product_taille_dF_over_dB; p++)
     {
	std::cout<<"dF_over_dB["<<p<<"] = "<<dF_over_dB[p]<<std::endl;
//        printf("p =  %d et dF_over_dB = %f\n",p,dF_over_dB[p]);
     }
*/

//repère_cuda °


// v y x 3*i+.
//std::cout<<"DEBUG 1"<<std::endl;
#pragma omp parallel private(k,l,j) shared(dF_over_dB_bis,params_T,M,indice_v,indice_y,indice_x)
{
#pragma omp for
for(int k=0; k<indice_v; k++){
	for(int l=0; l<indice_y; l++){
		for(int j=0; j<indice_x; j++){
			for(int i=0; i<M.n_gauss; i++){
				//ojqpsfdijpsqojdfpoidspoidfpojqspodij sdjfpjqspodfijiopqs jdpofijojfoiq sdoipf oqisdhfophjqsopidf opis dofi poqs dfop qspoidf jopiqjsdopif jopqisd fop Âsopid fpoij qsdf
				double par0 = params_T[l][j][3*i+0];
				double par1 = double(k+1) - params_T[l][j][3*i+1];
				double par2 = params_T[l][j][3*i+2];

//				double par1_k = double(k+1) - par1;
				double par2_pow = 1/(2*pow(par2,2.));
//dF_over_dB[v][y][x][M.n__gauss]
				dF_over_dB_bis[k][l][j][3*i+0] = exp( -pow( par1 ,2.)*par2_pow );//( -pow( par1_k ,2.)*par2_pow );
				dF_over_dB_bis[k][l][j][3*i+1] = par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow ); //par0*par1_k*par2_pow*2 * exp(-pow( par1_k,2.)*par2_pow );
				dF_over_dB_bis[k][l][j][3*i+2] = par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow );//par0*pow( par1_k, 2.)/(pow(par2,3.)) * exp(-pow( par1_k,2.)*par2_pow );
//				std::cout<<"par0 = "<<par0<<std::endl;
			}
		}
	}
}
}

/*
for(int k=0; k<10; k++){
//if (k<2){
	for(int l=0; l<indice_y; l++){
		for(int j=0; j<indice_x; j++){
			for(int i=0; i<3*M.n_gauss; i++){
	std::cout<<"dF_over_dB_bis["<<k<<"]["<<l<<"]["<<j<<"]["<< i <<"] ="<< dF_over_dB_bis[k][l][j][i]<<std::endl;
	std::cout<<"dF_over_dB    ["<<k<<"]["<<l<<"]["<<j<<"]["<< i <<"] ="<< dF_over_dB[i+j*t3+l*t2*t3+k*t2*t1*t3]<<std::endl;
	std::cout<<"  "<<std::endl;
			}
		}
	}
//}
}
*/



/*
for(int k=0; k<indice_v; k++){
if (k==0){
	for(int l=0; l<indice_y; l++){
		for(int j=0; j<indice_x; j++){
			for(int i=0; i<3*M.n_gauss; i++){
	std::cout<<"dF_over_dB_bis["<<k<<"]["<<l<<"]["<<j<<"]["<< i <<"] ="<< dF_over_dB_bis[k][l][j][i]<<std::endl;
			}
		}
	}
}
}
std::cout<<"dF_over_dB_bis[1][0][0][0] ="<< dF_over_dB_bis[1][0][0][0]<<std::endl;
std::cout<<"dF_over_dB_bis[0][1][0][0] ="<< dF_over_dB_bis[0][1][0][0]<<std::endl;
std::cout<<"dF_over_dB_bis[0][0][1][0] ="<< dF_over_dB_bis[0][0][1][0]<<std::endl;
std::cout<<"dF_over_dB_bis[0][0][0][1] ="<< dF_over_dB_bis[0][0][0][1]<<std::endl;


exit(0);
*/
double temps2_dF_dB = omp_get_wtime();

double temps1_deriv = omp_get_wtime();
//Bon code deriv
/*
#pragma omp parallel private(k,l) shared(deriv,dF_over_dB,M,indice_v,indice_y,indice_x,std_map,residual)
{
#pragma omp for
for(k=0; k<indice_v; k++){
	for(l=0; l<3*M.n_gauss; l++){
		for(i=0; i<indice_y; i++){
			for(j=0; j<indice_x; j++){
				if(std_map[i][j]>0.){
					deriv[l][i][j]+=  dF_over_dB[l][k][i][j]*residual[j][i][k]/pow(std_map[i][j],2);
				}
			}
		}
	}
}
std::cout<<"2 : "<<params[3*M.n_gauss-1][indice_y-1][indice_x-1]<<std::endl;
std::cout<<"1 : "<<params_T[indice_y-1][indice_x-1][3*M.n_gauss-1]<<std::endl;

}
*/

//$$
// DERIV pour dF_over_dB_bis
	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
			for(k=0; k<indice_v; k++){
				for(l=0; l<3*M.n_gauss; l++){
				if(std_map[i][j]>0.){
					deriv[l][i][j]+=  dF_over_dB_bis[k][i][j][l]*residual[j][i][k]/pow(std_map[i][j],2);
					}
				}
			}
		}
	}



/*
#pragma omp parallel private(i) shared(deriv, dF_over_dB, residual, std_map,M,indice_v,indice_y,indice_x)
{
#pragma omp for
*/

/*
	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
			for(k=0; k<indice_v; k++){
				for(l=0; l<3*M.n_gauss; l++){
				if(std_map[i][j]>0.){
//					coordinates[0] = l;
//					coordinates[1] = k;
//					coordinates[2] = i;
//					coordinates[3] = j;
					deriv[l][i][j]+=  dF_over_dB[k*t2*t1*t3+i*t2*t3+j*t3+l]*residual[j][i][k]/pow(std_map[i][j],2);
				}
			}
		}
	}
}
*/

/*
for(int j=0; j<indice_x; j++){
	for(int i=0; i<indice_y; i++){
		for(int l=0; l<3*M.n_gauss; l++){
	std::cout<<"deriv_bis["<<l<<"]["<<i<<"]["<< j <<"] ="<< deriv_bis[l][i][j]<<std::endl;
	std::cout<<"deriv    ["<<l<<"]["<<i<<"]["<< j <<"] ="<< deriv[l][i][j]<<std::endl;
	std::cout<<"  "<<std::endl;
		}
	}
}
*/




/*
		for(i=0; i<indice_y; i++){

			for(k=0; k<indice_v; k++){
	for(j=0; j<indice_x; j++){
			    for(l=0; l<3*M.n_gauss; l++){

			         if(std_map[i][j]>0.){
					deriv[l][i][j]+=  dF_over_dB[k][i][j][l]*residual[j][i][k]/pow(std_map[i][j],2);
					//dF_over_dB[v][y][x][M.n__gauss]
					//deriv[gauss][y][x]
				}
				}
			}
		}
	}
*/
//}
// v y x 3*i+.

double temps2_deriv = omp_get_wtime();

double temps1_conv = omp_get_wtime();

for(int k=0; k<M.n_gauss; k++){

	for(int p=0; p<indice_y; p++){
		for(int q=0; q<indice_x; q++){
			image_amp[p][q]=params[0+3*k][p][q];
			image_mu[p][q]=params[1+3*k][p][q];
			image_sig[p][q]=params[2+3*k][p][q];
		}
	}

	convolution_2D_mirror(M, image_amp, conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, image_mu, conv_mu, indice_y, indice_x,3);
	convolution_2D_mirror(M, image_sig, conv_sig, indice_y, indice_x,3);

	convolution_2D_mirror(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);



	for(int j=0; i<indice_y; j++){
		for(int i=0; j<indice_x; i++){
			f+= 0.5*M.lambda_amp*pow(conv_amp[i][j],2) + 0.5*M.lambda_var_amp*pow(image_amp[i][j]-mean_amp[k],2);
			f+= 0.5*M.lambda_mu*pow(conv_mu[i][j],2) + 0.5*M.lambda_var_mu*pow(image_mu[i][j]-mean_mu[k],2);
			f+= 0.5*M.lambda_sig*pow(conv_sig[i][j],2) + 0.5*M.lambda_var_sig*pow(image_sig[i][j]-mean_sig[k],2);

			dR_over_dB[0+3*k][i][j] = M.lambda_amp*conv_conv_amp[i][j]+M.lambda_var_amp*(image_amp[i][j]-mean_amp[k]);
			dR_over_dB[1+3*k][i][j] = M.lambda_mu*conv_conv_mu[i][j]+M.lambda_var_mu*(image_mu[i][j]-mean_mu[k]);
			dR_over_dB[2+3*k][i][j] = M.lambda_sig*conv_conv_sig[i][j]+M.lambda_var_sig*(image_sig[i][j]-mean_sig[k]);
		}
	}
}

	for(int l=0; l<3*M.n_gauss; l++){
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				g_3D[l][i][j] = deriv[l][i][j] + dR_over_dB[l][i][j];
			}
		}//deriv 3*M.n_gauss y x
	}
	double temps2_conv = omp_get_wtime();

	temps1_ravel = omp_get_wtime();
	ravel_3D(g_3D, g, 3*M.n_gauss, indice_y, indice_x);
	temps2_ravel = omp_get_wtime();
	temps_ravel+=temps2_ravel-temps1_ravel;

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;
	temps_ravel+=temps2_ravel-temps1_ravel;

	temps_f_g_cube += temps2_dF_dB - temps1_dF_dB;

}

void algo_rohsa::f_g_cube_omp_without_regul(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig) 
{

//std::cout<<"Début f_g_cube_test"<<std::endl;

std::vector<std::vector<std::vector<double>>> dR_over_dB(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
//°
//décommenter en bas
std::vector<std::vector<std::vector<std::vector<double>>>> dF_over_dB_bis(indice_v,std::vector<std::vector<std::vector<double>>>( indice_y,std::vector<std::vector<double>>(indice_x, std::vector<double>(3*M.n_gauss,0.))));
std::vector<std::vector<std::vector<double>>> deriv(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
//std::vector<std::vector<std::vector<double>>> deriv_bis(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<std::vector<std::vector<double>>> g_3D(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
//dF_over_dB[v][y][x][M.n__gauss]
//deriv[gauss][y][x]
std::vector<std::vector<std::vector<double>>> residual(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_v,0.)));

std::vector<std::vector<std::vector<double>>> params(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<std::vector<std::vector<double>>> params_T(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(3*M.n_gauss,0.)));
std::vector<double> b_params(M.n_gauss,0.);

std::vector<std::vector<double>> conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> conv_conv_sig(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_amp(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_mu(indice_y,std::vector<double>(indice_x, 0.));
std::vector<std::vector<double>> image_sig(indice_y,std::vector<double>(indice_x, 0.));

int n_beta = (3*M.n_gauss*indice_x*indice_y);//+M.n_gauss;

int i,k,j,l,p;

for(int i = 0; i< n; i++){
	g[i]=0.;
}
f=0.;

double temps1_ravel = omp_get_wtime();

//extracting one params_T for the gradient and another for calculations below
// WARNING
unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

unravel_3D_T(beta, params_T,  3*M.n_gauss, indice_x,indice_y);

int N_x = indice_x;
int N_y = indice_y;
int N_z = 3*M.n_gauss;

/*
std::cout<<"1 : "<<params_T[indice_y-1][indice_x-1][3*M.n_gauss-1]<<std::endl;

std::cout<<"2 : "<<params[3*M.n_gauss-1][0][1]<<std::endl;
std::cout<<"1 : "<<params_T[0][1][3*M.n_gauss-1]<<std::endl;
*/
//exit(0);
double temps2_ravel = omp_get_wtime();
temps_ravel+=temps2_ravel-temps1_ravel;

double temps1_tableaux = omp_get_wtime();
for(j=0; j<indice_x; j++){
	for(i=0; i<indice_y; i++){
		std::vector<double> residual_1D(indice_v,0.);
		std::vector<double> params_flat(params.size(),0.);
		std::vector<double> cube_flat(cube[0][0].size(),0.);
		for (p=0;p<params_flat.size();p++){
			params_flat[p]=params[p][i][j];
		}
		for (p=0;p<cube_flat.size();p++){
			cube_flat[p]=cube[i][j][p];
		}
		myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);
		for (p=0;p<residual_1D.size();p++){
			residual[j][i][p]=residual_1D[p];
		}
		if(std_map[i][j]>0.){
			f+=myfunc_spec(residual_1D)/pow(std_map[i][j],2.); //std_map est arrondie... 
		}
	}

//}
}

double temps2_tableaux = omp_get_wtime();

double temps1_dF_dB = omp_get_wtime();


// v y x 3*i+.
//std::cout<<"DEBUG 1"<<std::endl;
#pragma omp parallel private(k,l,j) shared(dF_over_dB_bis,params_T,M,indice_v,indice_y,indice_x)
{
#pragma omp for
for(int k=0; k<indice_v; k++){
	for(int l=0; l<indice_y; l++){
		for(int j=0; j<indice_x; j++){
			for(int i=0; i<M.n_gauss; i++){
				//ojqpsfdijpsqojdfpoidspoidfpojqspodij sdjfpjqspodfijiopqs jdpofijojfoiq sdoipf oqisdhfophjqsopidf opis dofi poqs dfop qspoidf jopiqjsdopif jopqisd fop Âsopid fpoij qsdf
				double par0 = params_T[l][j][3*i+0];
				double par1 = double(k+1) - params_T[l][j][3*i+1];
				double par2 = params_T[l][j][3*i+2];

//				double par1_k = double(k+1) - par1;
				double par2_pow = 1/(2*pow(par2,2.));
//dF_over_dB[v][y][x][M.n__gauss]
				dF_over_dB_bis[k][l][j][3*i+0] = exp( -pow( par1 ,2.)*par2_pow );//( -pow( par1_k ,2.)*par2_pow );
				dF_over_dB_bis[k][l][j][3*i+1] = par0*par1*par2_pow*2 * exp(-pow( par1,2.)*par2_pow ); //par0*par1_k*par2_pow*2 * exp(-pow( par1_k,2.)*par2_pow );
				dF_over_dB_bis[k][l][j][3*i+2] = par0*pow( par1, 2.)/(pow(par2,3.))*exp(-pow(par1 ,2.)*par2_pow );//par0*pow( par1_k, 2.)/(pow(par2,3.)) * exp(-pow( par1_k,2.)*par2_pow );
//				std::cout<<"par0 = "<<par0<<std::endl;
			}
		}
	}
}
}

double temps2_dF_dB = omp_get_wtime();

double temps1_deriv = omp_get_wtime();

	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
			for(k=0; k<indice_v; k++){
				for(l=0; l<3*M.n_gauss; l++){
				if(std_map[i][j]>0.){
					deriv[l][i][j]+=  dF_over_dB_bis[k][i][j][l]*residual[j][i][k]/pow(std_map[i][j],2);
					}
				}
			}
		}
	}



double temps2_deriv = omp_get_wtime();

double temps1_conv = omp_get_wtime();

	for(int l=0; l<3*M.n_gauss; l++){
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				g_3D[l][i][j] = deriv[l][i][j];
			}
		}//deriv 3*M.n_gauss y x
	}
	double temps2_conv = omp_get_wtime();

	temps1_ravel = omp_get_wtime();
	ravel_3D(g_3D, g, 3*M.n_gauss, indice_y, indice_x);
	temps2_ravel = omp_get_wtime();
	temps_ravel+=temps2_ravel-temps1_ravel;

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;
	temps_ravel+=temps2_ravel-temps1_ravel;

	temps_f_g_cube += temps2_dF_dB - temps1_dF_dB;

}




void algo_rohsa::f_g_cube_naive(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig){

//std::cout<<"Début f_g_cube_test"<<std::endl;
	std::vector<std::vector<std::vector<double>>> deriv(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
	std::vector<std::vector<std::vector<double>>> g_3D(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));

	std::vector<std::vector<std::vector<double>>> residual(indice_x,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_v,0.)));

	std::vector<std::vector<std::vector<double>>> params(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
	std::vector<double> b_params(M.n_gauss,0.);

	std::vector<std::vector<std::vector<double>>> dR_over_dB(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
	std::vector<std::vector<std::vector<std::vector<double>>>> dF_over_dB(3*M.n_gauss,std::vector<std::vector<std::vector<double>>>( indice_v,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.))));

	std::vector<std::vector<double>> conv_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_sig(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> conv_conv_sig(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_amp(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_mu(indice_y,std::vector<double>(indice_x, 0.));
	std::vector<std::vector<double>> image_sig(indice_y,std::vector<double>(indice_x, 0.));

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;

	for(int i = 0; i< n; i++){
		g[i]=0.;
	}
	f=0.;

	unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

	for(int j=0; j<indice_x; j++){
		for(int i=0; i<indice_y; i++){
			std::vector<double> residual_1D(indice_v,0.);
			std::vector<double> params_flat(params.size(),0.);
			std::vector<double> cube_flat(cube[0][0].size(),0.);

			for (int p=0;p<params_flat.size();p++){
				params_flat[p]=params[p][i][j];
			}

//			std::cout<< "--> cube_flat.size() = "<<cube_flat.size() <<std::endl;
			for (int p=0;p<cube_flat.size();p++){
				cube_flat[p]=cube[j][i][p];
			}

			myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);

			for (int p=0;p<residual_1D.size();p++){
				residual[j][i][p]=residual_1D[p];
			}

			if(std_map[i][j]>0.){
				f += myfunc_spec(residual_1D)*1/pow(std_map[i][j],2.); //std_map est arrondie... 
			}
		}
	}
//§§
//	printf("->f = %f\n",f);
//	std::cout<<"->f = "<<f<<std::endl;
/*
	for(int j=0; j<indice_y; j++){
		for(int i = 0; i<indice_x;i++){
		std::cout<<"std_map["<<i<<"]["<<j<<"] = "<<std_map[i][j]<<std::endl;
		}
	}
*/
//	exit(0);

//	std::cout<< "AVANT GRADIENT+CONVO" <<std::endl;

	for(int i=0; i<M.n_gauss; i++){
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){
				image_amp[p][q]=params[0+3*i][p][q];
				image_mu[p][q]=params[1+3*i][p][q];
				image_sig[p][q]=params[2+3*i][p][q];
			}
		}

//		std::cout<< "AVANT CONVO" <<std::endl;

		convolution_2D_mirror(M, image_amp, conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror(M, image_mu, conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror(M, image_sig, conv_sig, indice_y, indice_x,3);
	
		convolution_2D_mirror(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);

//		std::cout<< "AVANT SOMME F" <<std::endl;
		for(int l=0; l<indice_x; l++){
			for(int j=0; j<indice_y; j++){
				f+= 0.5*M.lambda_amp*pow(conv_amp[j][l],2);
				f+= 0.5*M.lambda_mu*pow(conv_mu[j][l],2);
				f+= 0.5*M.lambda_sig*pow(conv_sig[j][l],2)+0.5*M.lambda_var_sig*pow(image_sig[j][l]-b_params[i],2);

				g[n_beta-M.n_gauss+i] += M.lambda_var_sig*(b_params[i]-image_sig[j][l]);

				deriv[0+3*i][j][l] += M.lambda_amp*conv_conv_amp[j][l];
				deriv[1+3*i][j][l] += M.lambda_mu*conv_conv_mu[j][l];
				deriv[2+3*i][j][l] += M.lambda_sig*conv_conv_sig[j][l]+M.lambda_var_sig*(image_sig[j][l]-b_params[i]);
			}
		}
	}
	for(int i=0; i<M.n_gauss; i++){
		for(int l=0; l<indice_x; l++){
			for(int j=0; j<indice_y; j++){
			int k;
			double v_1=0.,v_2=0.,v_3=0.;

		//		std::cout<< "AVANT GRADIENT BIDOUILLÉ" <<std::endl;

			if(std_map[j][l]>0.){
				for(k=0; k<indice_v; k++){
					v_1 += exp(-pow( double(k)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
			//printf("--->curseur = %e\n", std_map[j][l] );//residual[l][j][k]  );
					v_2 +=  params[3*i][j][l]*( double(k) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( double(k)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
				
					v_3 += params[3*i][j][l]*pow( double(k) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( double(k)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
					}
				}
			deriv[0+3*i][j][l]+=v_1;
			deriv[1+3*i][j][l]+=v_2;
			deriv[2+3*i][j][l]+=v_3;
			}
		}
	}

	ravel_3D(deriv, g, 3*M.n_gauss, indice_y, indice_x);
}


void algo_rohsa::minimize(parameters &M, long n, long m, std::vector<double> &beta, std::vector<double> &lb_v, std::vector<double> &ub_v, std::vector<std::vector<std::vector<double>>> &cube, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig, int dim_x, int dim_y, int dim_v, double* cube_flattened) {
    /* System generated locals */
    int i__1;
	int  i__c = 0;
    double d__1, d__2;
    /* Local variables */

    double t1, t2, f;// , g[n];
//	std::cout<< "AVANT MALLOC G AVEC n" <<std::endl;

	double* g = (double*)malloc(n*sizeof(double));
//	std::cout<< "APRÈS MALLOC G AVEC n" <<std::endl;

    int i__;
    int taille_wa = 2*M.m*n+5*n+11*M.m*M.m+8*M.m;
    int taille_iwa = 3*n;


    long* nbd = NULL;
    nbd = (long*)malloc(n*sizeof(double)); 
    long* iwa = NULL;
    iwa = (long*)malloc(taille_iwa*sizeof(double)); 

//    long nbd[n], iwa[taille_iwa];
/*
int* memoireAllouee = NULL; // On crée un pointeur sur int

memoireAllouee = malloc(sizeof(int)); // La fonction malloc inscrit dans notre pointeur l'adresse qui a été reservée.
*/
    double* wa = NULL;
    wa = (double*)malloc(taille_wa*sizeof(double)); 
//    double wa[taille_wa];

//std::cout << " DEBUG  TEST 1" << std::endl;
/*     char task[60]; */
    long taskValue;

    long *task=&taskValue; /* must initialize !! */
/*      http://stackoverflow.com/a/11278093/269192 */
    double factr;
    long csaveValue;

    long *csave=&csaveValue;
//std::cout << " DEBUG  TEST -1" << std::endl;
    double dsave[29];
    long isave[44];
    logical lsave[4];
    double pgtol;


//std::cout << " DEBUG  TEST 2" << std::endl;
/*
std::cout << "beta.size() = " <<beta.size()<< std::endl;
std::cout << "lb_v.size() = " <<lb_v.size()<< std::endl;
std::cout << "ub_v.size() = " <<ub_v.size()<< std::endl;
*/
// converts the vectors into a regular list
//2Bfixed
//std::cout << " DEBUG  TEST 0" << std::endl;
	double* x = (double*)malloc(beta.size()*sizeof(double));
	double* lb = (double*)malloc(lb_v.size()*sizeof(double));
	double* ub = (double*)malloc(ub_v.size()*sizeof(double));

	double temps2_tableau_update = omp_get_wtime();

    for(int i(0); i<beta.size(); i++) {
	x[i]=beta[i];
    } 
    for(int i(0); i<lb_v.size(); i++) {
	lb[i]=lb_v[i];
    } 
    for(int i(0); i<ub_v.size(); i++) {
	ub[i]=ub_v[i];
    } 

//	std::cout<< "AVANT BOUCLE i<n" <<std::endl;

    for(int i(0); i<n; i++) {
	g[i]=0.;
    } 
//	std::cout<< "APRÈS BOUCLE i<n" <<std::endl;
	temps_tableau_update += omp_get_wtime() - temps2_tableau_update;

     f=0.;

/*     We specify the tolerances in the stopping criteria. */
    factr = 1e7;
    pgtol = 1e-5;

/*     We now provide nbd which defines the bounds on the variables: */
/*                    l   specifies the lower bounds, */
/*                    u   specifies the upper bounds. */
/*     First set bounds on the odd-numbered variables. */

//std::cout << " DEBUG  TEST 3" << std::endl;
    for (i__ = 0; i__ < n; i__ ++) {
        nbd[i__] = 2;
    }

    /*     We start the iteration by initializing task. */
    *task = (long)START;
/*     s_copy(task, "START", (ftnlen)60, (ftnlen)5); */
    /*        ------- the beginning of the loop ---------- */
L111:

	double temps1_f_g_cube = omp_get_wtime();
//std::cout << " DEBUG  TEST 4" << std::endl;
    while(IS_FG(*task) or *task==NEW_X or *task==START){ 
		i__c ++;
    /*     This is the call to the L-BFGS-B code. */
//    std::cout<<" Début appel BFGS "<<std::endl;
////printf("g[0] = %f \n",g[0] );
////printf("f = %f \n",f );
////printf("x[0] = %f \n",x[0] );
////printf("cube_flattened[0] = %f \n",cube_flattened[0] );
////printf("lb[0] = %f \n",lb[0] );
////printf("ub[0] = %f \n",ub[0] );
////printf("n = %d	\n",int(n) );

////printf("setulb \n");

	double temps_temp = omp_get_wtime();
    setulb(&n, &m, x, lb, ub, nbd, &f, g, &factr, &pgtol, wa, iwa, task, 
            &M.iprint, csave, lsave, isave, dsave);
	temps_setulb += omp_get_wtime() - temps_temp;

/*     if (s_cmp(task, "FG", (ftnlen)2, (ftnlen)2) == 0) { */
    if ( IS_FG(*task) ) {

//µ$£ù
//either f_g_cube or f_g_cube_vector or f_g_cube_test
// Wrong values
//	f_g_cube_old_archive(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig);
//	f_g_cube_omp(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig);

// good values
// CPU
//	f_g_cube_vector(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig); // OK grandes données
//	f_g_cube_naive(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig);
//	f_g_cube_fast(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig); //OK

// GPU

//	f_g_cube_cuda_L(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig, cube_flattened); // expérimentation gradient
//	f_g_cube_cuda_L_2(M, f, g, n, cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig); // ça marche !! (c'est f_g_cube_cuda_L_2_2Bverified sans la partie vérification)
	f_g_cube_parallel(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig, cube_flattened, this->temps_conv, this->temps_deriv, this->temps_tableaux, this->temps_f_g_cube);


//exit(0);

////printf("g[0] = %f \n",g[0] );
////printf("f = %f \n",f );
////printf("x[0] = %f \n",x[0] );
////printf("cube_flattened[0] = %f \n",cube_flattened[0] );

////printf("f_g_cube \n");

/*
    for(int i(0); i<n; i++) {
		printf("x[%d] = %f \n", i, x[i]);
    } 
*/


/*
printf("g[0] = %f \n",g[0] );
printf("f = %f \n",f );
printf("x[0] = %f \n",x[0] );
*/
//printf("cube_flattened[0] = %f \n",cube_flattened[0] );

if (i__c >2){
//	exit(0);
}

/*
for(int i = 0; i<(3*M.n_gauss*dim_x*dim_y)+M.n_gauss; i++){
	printf("g[%d] = %f \n",i,g[i] );
}
printf("f = %f \n",f );
*/
//std::cin.ignore();



//	f_g_cube_cuda_L_deux_tiers(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig, cube_flattened); // doesn't work
//	f_g_cube_cuda_L_2_2Bverified(M, f, g, n, cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig); //gradient qui marche



// not precise


//	f_g_cube_fast_without_regul(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig); 
//	f_g_cube_sep(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig);
//	f_g_cube_test(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig);

/*
std::cout<< "Après gradient " <<std::endl;
   for(int p; p<4; p++)
     {
	std::cout<< " -> "<<g[p] <<std::endl;
	 }
	std::cout<< " f = "<<f <<std::endl;
std::cout<< "pas d'erreur ? " <<std::endl;
*/

//usual gradient with vectors
//	f_g_cube_vector(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, mean_amp, mean_mu, mean_sig);

//	std::cout<<"DEBUG 3"<<std::endl;
	}

	if (*task==NEW_X ) {
		if (isave[33] >= M.maxiter) {
			*task = STOP_ITER;
			}
		if (dsave[12] <= (fabs(f) + 1.) * 1e-10) {
			*task = STOP_GRAD;
		}
	}

	}

/*
	printf("x[0] = %f \n",x[0] );
	printf("x[1] = %f \n",x[1] );
	printf("x[2] = %f \n",x[2] );
	exit(0);
*/

//std::cout << " DEBUG  TEST 3" << std::endl;
	double temps4_tableau_update = omp_get_wtime();
	
	for(int i(0); i<beta.size(); i++) {
		beta[i]=x[i];
//		printf("x[%d] = %f\n",i,x[i]);
	}
//	std::cin.ignore();
	temps_tableau_update += omp_get_wtime() - temps4_tableau_update;

//exit(0);
	double temps2_f_g_cube = omp_get_wtime();

	std::cout<< "Temps de calcul gradient : " << temps2_f_g_cube - temps1_f_g_cube<<std::endl;


    /* System generated locals */

	free(wa);
	free(nbd);
	free(iwa);
	free(x);
	free(lb);
	free(ub);
	free(g);

//	std::cout << " DEBUG  TEST 4" << std::endl;
/*
	free(g);
	free(nbd);
	free(iwa);

	free(dsave);
	free(isave);
	free(lsave);

*/
}




void algo_rohsa::go_up_level(std::vector<std::vector<std::vector<double>>> &fit_params) {
		//dimensions of fit_params
	int dim[3];
	dim[2]=fit_params[0][0].size();
	dim[1]=fit_params[0].size();
	dim[0]=fit_params.size();
 
	std::vector<std::vector<std::vector<double>>> cube_params_down(dim[0],std::vector<std::vector<double>>(dim[1], std::vector<double>(dim[2],0.)));	

	for(int i = 0; i<dim[0]	; i++){
		for(int j = 0; j<dim[1]; j++){
			for(int k = 0; k<dim[2]; k++){
				cube_params_down[i][j][k]=fit_params[i][j][k];
			}
		}
	}


	fit_params.resize(dim[0]);
	for(int i=0;i<dim[0];i++)
	{
	   fit_params[i].resize(2*dim[1]);
	   for(int j=0;j<2*dim[1];j++)
	   {
	       fit_params[i][j].resize(2*dim[2], 0.);
	   }
	}

	std::cout << "fit_params.size() : " << fit_params.size() << " , " << fit_params[0].size() << " , " << fit_params[0][0].size() <<  std::endl;


	for(int i = 0; i<dim[0]; i++){
		for(int j = 0; j<2*dim[1]; j++){
			for(int k = 0; k<2*dim[2]; k++){
				fit_params[i][j][k]=0.;
			}
		}
	}

	for(int i = 0; i<dim[1]; i++){
		for(int j = 0; j<dim[2]; j++){
			for(int k = 0; k<2; k++){
				for(int l = 0; l<2; l++){
					for(int m = 0; m<dim[0]; m++){
						fit_params[m][k+i*2][l+j*2] = cube_params_down[m][i][j];
					}
				}
			}
		}
	}
}

void algo_rohsa::upgrade(parameters &M, std::vector<std::vector<std::vector<double>>> &cube, std::vector<std::vector<std::vector<double>>> &params, int power) {
        int i,j;
//        int nb_threads = omp_get_max_threads();
//        printf(">> omp_get_max_thread()\n>> %i\n", nb_threads);

//      #pragma omp parallel shared(cube,params) shared(power) shared(M)
//       {
//        printf("thread:%d\n", omp_get_thread_num());
        std::vector<double> line(dim_v,0.);
        std::vector<double> x(3*M.n_gauss,0.);
        std::vector<double> lb(3*M.n_gauss,0.);
        std::vector<double> ub(3*M.n_gauss,0.);
        for(i=0;i<power; i++){
                for(j=0;j<power; j++){
                        int p;
                        for(p=0; p<cube[0][0].size();p++){
                                line[p]=cube[i][j][p];
                        }
                        for(p=0; p<params.size(); p++){
                                x[p]=params[p][i][j]; //cache
                        }
                        init_bounds(M, line, M.n_gauss, lb, ub, false); //bool _init = false;
                        minimize_spec(M,3*M.n_gauss ,M.m ,x ,lb , M.n_gauss, ub ,line);
                        for(p=0; p<params.size();p++){
                                params[p][i][j]=x[p]; //cache
//                              std::cout << "p = "<<p<<  std::endl;
                        }
                }
//        }
        }
}



void algo_rohsa::init_bounds(parameters &M, std::vector<double> line, int n_gauss_local, std::vector<double> &lb, std::vector<double> &ub, bool _init) {

	double max_line = *std::max_element(line.begin(), line.end());
//	std::cout<<"max_line = "<<max_line<<std::endl;
	for(int i(0); i<n_gauss_local; i++) {
		lb[0+3*i]=0.;
		ub[0+3*i]=max_line;

		lb[1+3*i]=0.;
		ub[1+3*i]=dim_v;
		if (_init){
			lb[2+3*i]=M.lb_sig_init;
			ub[2+3*i]=M.ub_sig_init;
		}else{
			lb[2+3*i]=M.lb_sig;
			ub[2+3*i]=M.ub_sig;
		}
	}
}

double algo_rohsa::model_function(int x, double a, double m, double s) {

	return a*exp(-pow((double(x)-m),2.) / (2.*pow(s,2.)));

}

int algo_rohsa::minloc(std::vector<double> &tab) {
	return std::distance(tab.begin(), std::min_element( tab.begin(), tab.end() ));
}

void algo_rohsa::minimize_spec(parameters &M, long n, long m, std::vector<double> &x_v, std::vector<double> &lb_v, int n_gauss_i, std::vector<double> &ub_v, std::vector<double> &line_v) {
/* Minimize_spec */ 
//int MAIN__(void)
    std::vector<double> _residual_;
    for(int p(0); p<dim_v; p++){
	_residual_.vector::push_back(0.);
    }
    /* System generated locals */
    int i__1;
    double d__1, d__2;
    /* Local variables */
    double f, g[n];
    int i__;

    int taille_wa = 2*m*n+5*n+11*m*m+8*m;
    int taille_iwa = 3*n;
    double t1, t2, wa[taille_wa];
    long nbd[n], iwa[taille_iwa];
/*     char task[60]; */
    long taskValue;
    long *task=&taskValue; /* must initialize !! */
/*      http://stackoverflow.com/a/11278093/269192 */
    double factr;
    long csaveValue;
    long *csave=&csaveValue;
    double dsave[29];
    long isave[44];
    logical lsave[4];
    double pgtol;

// converts the vectors into a regular list
    double tampon(0.);
    double x[x_v.size()];
    double lb[lb_v.size()];
    double ub[ub_v.size()];
    double line[line_v.size()];

    for(int i(0); i<line_v.size(); i++) {
	line[i]=line_v[i];
    } 

    for(int i(0); i<x_v.size(); i++) {
	x[i]=x_v[i];
    } 
    for(int i(0); i<lb_v.size(); i++) {
	lb[i]=lb_v[i];
    } 
    for(int i(0); i<ub_v.size(); i++) {
	ub[i]=ub_v[i];
    } 
/*     We specify the tolerances in the stopping criteria. */
    factr = 1e7;
    pgtol = 1e-5;

/*     We now provide nbd which defines the bounds on the variables: */
/*                    l   specifies the lower bounds, */
/*                    u   specifies the upper bounds. */
/*     First set bounds on the odd-numbered variables. */
    for (i__ = 0; i__ < n; i__ ++) {
        nbd[i__] = 2;
    }

    /*     We start the iteration by initializing task. */
    *task = (long)START;
/*     s_copy(task, "START", (ftnlen)60, (ftnlen)5); */
    /*        ------- the beginning of the loop ---------- */
L111:
    while(IS_FG(*task) or *task==NEW_X or *task==START){ 
    /*     This is the call to the L-BFGS-B code. */
//    std::cout<<" Début appel BFGS "<<std::endl;

    setulb(&n, &m, x, lb, ub, nbd, &f, g, &factr, &pgtol, wa, iwa, task, 
            &M.iprint, csave, lsave, isave, dsave);
//ùù
/*     if (s_cmp(task, "FG", (ftnlen)2, (ftnlen)2) == 0) { */
    if ( IS_FG(*task) ) {
		myresidual(x, line, _residual_, n_gauss_i);
		f = myfunc_spec(_residual_);
		mygrad_spec(g, _residual_, x, n_gauss_i);
	}


    if (*task==NEW_X ) {
	if (isave[33] >= M.maxiter) {
		*task = STOP_ITER;
		}
	
	if (dsave[12] <= (fabs(f) + 1.) * 1e-10) {
		*task = STOP_GRAD;
	}
     }


        /*          go back to the minimization routine. */
//if (compteurX<100000000){        
//	goto L111;
//}

	}
	for(int i(0); i<x_v.size(); i++) {
		x_v[i]=x[i];
//		std::cout<<"x["<<i<<"] = "<<x[i]<<std::endl;
	}
//exit(0);

}

double algo_rohsa::myfunc_spec(std::vector<double> &residual) {
	double S(0.);
	for(int p(0); p<residual.size(); p++) {
		S+=pow(residual[p],2);
	}
	return 0.5*S;
}

void algo_rohsa::myresidual(double params[], double line[], std::vector<double> &residual, int n_gauss_i) {
	int i,k;
	std::vector<double> model(residual.size(),0.);
//	#pragma omp parallel private(i,k) shared(params)
//	{
//	#pragma omp for
	for(i=0; i<n_gauss_i; i++) {
		for(k=0; k<residual.size(); k++) {
			model[k]+= model_function(k+1, params[3*i], params[1+3*i], params[2+3*i]);
		}
	}
//	}
	for(int p=0; p<residual.size(); p++) {
		residual[p]=model[p]-line[p]; 
	}
}

void algo_rohsa::myresidual(std::vector<double> &params, std::vector<double> &line, std::vector<double> &residual, int n_gauss_i) {
	int k;
	std::vector<double> model(residual.size(),0.);

	for(int i(0); i<n_gauss_i; i++) {
		for(k=0; k<residual.size(); k++) {
			model[k]+= model_function(k, params[3*i], params[1+3*i], params[2+3*i]);
		}
	}

	for(int p=0; p<residual.size(); p++) {
		residual[p]=model[p]-line[p]; 
	}

}
void algo_rohsa::mygrad_spec(double gradient[], std::vector<double> &residual, double params[], int n_gauss_i) {

	std::vector<std::vector<double>> dF_over_dB(3*n_gauss_i, std::vector<double>(dim_v,0.));
	double g(0.);
	int i,k;
	for(int p(0); p<3*n_gauss_i; p++) {
		gradient[p]=0.;
	}

//	#pragma omp parallel num_threads(2) shared(dF_over_dB,params)
//	{
//	#pragma omp for private(i)
	for(i=0; i<n_gauss_i; i++) {
		for(int k(0); k<dim_v; k++) {
			dF_over_dB[0+3*i][k] += exp(-pow( double(k+1)-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

			dF_over_dB[1+3*i][k] +=  params[3*i]*( double(k+1) - params[1+3*i])/pow(params[2+3*i],2.) * exp(-pow( double(k+1)-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

			dF_over_dB[2+3*i][k] += params[3*i]*pow( double(k+1) - params[1+3*i] , 2.)/(pow(params[2+3*i],3.)) * exp(-pow( double(k+1)-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

		}
//	}
	}
//	#pragma omp parallel num_threads(2) shared(dF_over_dB, residual ,gradient)
//	{
//	#pragma omp for private(k)
	for(k=0; k<3*n_gauss_i; k++){
		for(int i=0; i<dim_v; i++){
			gradient[k]+=dF_over_dB[k][i]*residual[i];
	//		std::cout<<"dF_over_dB["<<k<<"]["<<i<<"] = "<< dF_over_dB[k][i]<<std::endl;
		}

//	}
	}
}


void algo_rohsa::init_spectrum(parameters &M, std::vector<double> &line, std::vector<double> &params) {

	
	int i;

/*
	for(i=0; i<dim_v; i++) {
		printf("line[%d] = %f \n", i,line[i]);
	}
	for(i=0; i<M.n_gauss; i++) {
		printf("params[%d] = %f \n", i,params[i]);
	}
*/

	for(i=1; i<=M.n_gauss; i++) {
		std::vector<double> model_tab(dim_v,0.);
		std::vector<double> residual(dim_v,0.);
		std::vector<double> lb(3*i,0.);
		std::vector<double> ub(3*i,0.);
		int rang = std::distance(residual.begin(), std::min_element( residual.begin(), residual.end() ));

		init_bounds(M, line,i,lb,ub, true); //we consider {bool _init = true;} since we want to initialize the boundaries
/*
		printf("line[0] = %f \n", line[0]);
		printf("lb[0] = %f \n", lb[0]);
		printf("ub[0] = %f \n", ub[0]);
*/
		for(int j(0); j<i; j++) {

			for(int k=0; k<dim_v; k++) {			
				model_tab[k]+= model_function(k+1,params[3*j], params[1+3*j], params[2+3*j]);
			}
		}
//		printf("model_tab[0] = %f \n", model_tab[0]);
		
		for(int p(0); p<dim_v; p++) {	
			residual[p]=model_tab[p]-line[p];
		}

/*
		for(int p(0); p<dim_v; p++) {	
			printf("residual[%d] = %f",p, residual[p]);
		}
*/
/*
		printf("minloc(residual) = %d \n", minloc(residual));
*/		
		std::vector<double> x(3*i,0.);

		for(int p(0); p<3*(i); p++){	
			x[p]=params[p];
		}
		
		x[1+3*(i-1)] = minloc(residual)+1;
		x[0+3*(i-1)] = line[int(x[1+3*(i-1)])-1]*M.amp_fact_init;
		x[2+3*(i-1)] = M.sig_init;

/*
		printf("x[0] = %f \n", x[0]);
		printf("x[1] = %f \n", x[1]);
		printf("x[2] = %f \n", x[2]);
*/

		minimize_spec(M, 3*i, M.m, x, lb, i, ub, line);

/*
		printf("x[0] = %f \n", x[0]);
		printf("x[1] = %f \n", x[1]);
		printf("x[2] = %f \n", x[2]);
*/

/*
		for(int k=0; k<params.size(); k++) {			
			printf("params[%d] = %f \n", k, params[k]);
		}

		for(int k=0; k<dim_v; k++) {			
			printf("model_tab[%d] = %f \n", k, model_tab[k]);
		}
		for(int k=0; k<dim_v; k++) {			
			printf("residual[%d] = %f \n", k, residual[k]);
		}
*/

//		exit(0);

		/*
		std::cout<<"lb = "<<std::endl;
		for(int c=0;c<lb.size();c++)
		{
			std::cout<<lb[c]<<std::endl;
		}
			std::cout<<"  "<<std::endl;

		std::cout<<"ub = "<<std::endl;
		for(int c=0;c<ub.size();c++)
		{
			std::cout<<ub[c]<<std::endl;
		}
			std::cout<<"  "<<std::endl;

		std::cout<<"x = "<<std::endl;
		for(int c=0;c<x.size();c++)
		{
			std::cout<<x[c]<<std::endl;
		}
			std::cout<<"  "<<std::endl;
		*/

		for(int p(0); p<3*(i); p++) {
			params[p] = x[p];
		}
	}

}




void algo_rohsa::mean_array(int power, std::vector<std::vector<std::vector<double>>> &cube_mean)
{
	std::vector<double> spectrum(file.dim_cube[2],0.);
	int n = file.dim_cube[1]/power;
	for(int i(0); i<cube_mean[0].size(); i++)
	{
		for(int j(0); j<cube_mean.size(); j++)
		{
			for(int k(0); k<n; k++)
			{
				for (int l(0); l<n; l++)
				{
					for(int m(0); m<file.dim_cube[2]; m++)
					{

//						std::cout<< "  test __  i,j,k,l,m,n ="<<i<<","<<j<<","<<k <<","<<l<<","<<m<<","<<n<< std::endl;
//						std::cout << "  test__ "<<k+j*n<<std::endl;
						spectrum[m] += file.cube[l+i*n][k+j*n][m];
					}
				}
			}
			for(int m(0); m<file.dim_cube[2]; m++)
			{
				cube_mean[j][i][m] = spectrum[m]/pow(n,2);
			}
			for(int p(0); p<file.dim_cube[2	]; p++)
			{
				spectrum[p] = 0.;
			}
		}
	}


}

void algo_rohsa::convolution_2D_mirror_flat(const parameters &M, double* image, double* &conv, int dim_y, int dim_x, int dim_k)
{
	int ii(0),jj(0),mm(0),nn(0),kCenterY(0), kCenterX(0);

	std::vector <std::vector<double>> ext_conv(dim_y+4, std::vector<double>(dim_x+4,0.));
	std::vector <std::vector<double>> extended(dim_y+4, std::vector<double>(dim_x+4,0.));

	float temps_temp = omp_get_wtime();

	for(int j(0); j<dim_x; j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			extended[2+i][2+j]=image[i+dim_x*j];
		}
	}

	for(int j(0); j<2; j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			extended[2+i][j] = image[i+dim_x*j];
		}
	}

	for(int i(0); i<2; i++)
	{
		for(int j(0); j<dim_x; j++)
		{
			extended[i][2+j] = image[i+dim_x*j];
		}
	}

	for(int j=dim_x; j<dim_x+2; j++)
	{
		for(int i=0; i<dim_y; i++)
		{
			extended[2+i][2+j]=image[i+dim_x*(j-2)];
		}
	}

	for(int j(0); j<dim_x; j++)
	{
		for(int i(dim_y); i<dim_y+2; i++)
		{
			extended[2+i][2+j]=image[i-2+dim_x*j];
		}
	}
	temps_mirroirs += omp_get_wtime() - temps_temp;
/*
	for(int j=0; j<dim_x+4; j++)
	{
		for(int i=0; i<dim_y+4; i++)
		{
			std::cout<<"extended["<<i+(dim_x+4)*j<<"] = "<<extended[i][j]<<std::endl;
		}
	}
*/

	kCenterY = dim_k/2+1;
	kCenterX = kCenterY;

	for(int j(1);j<=dim_x+4;j++)
	{
		for(int i(1); i<=dim_y+4; i++)
		{
			for(int m(1); m<=dim_k ; m++)
			{
				mm = dim_k - m + 1;

				for(int n(1);n<=dim_k;n++)
				{
					nn = dim_k - n + 1;

					ii = i + (m - kCenterY);
					jj = j + (n - kCenterX);

					if( ii >= 1 && ii < dim_y+4 && jj>=1 && jj< dim_x+4 )
					{
						ext_conv[i-1][j-1] += extended[ii-1][jj-1]*M.kernel[mm-1][nn-1];
					}
				}
			}
		}
	}

	for(int j(0);j<dim_x;j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			conv[i+dim_x*j] = ext_conv[2+i][2+j];
		}
	}
}

void algo_rohsa::convolution_2D_mirror_flat(const parameters &M, float* image, float* &conv, int dim_y, int dim_x, int dim_k, float temps_transfert, float temps_mirroirs)
{
	int ii(0),jj(0),mm(0),nn(0),kCenterY(0), kCenterX(0);

	std::vector <std::vector<float>> ext_conv(dim_y+4, std::vector<float>(dim_x+4,0.));
	std::vector <std::vector<float>> extended(dim_y+4, std::vector<float>(dim_x+4,0.));

	for(int j(0); j<dim_x; j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			extended[2+i][2+j]=image[i+dim_x*j];
		}
	}

	for(int j(0); j<2; j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			extended[2+i][j] = image[i+dim_x*j];
		}
	}

	for(int i(0); i<2; i++)
	{
		for(int j(0); j<dim_x; j++)
		{
			extended[i][2+j] = image[i+dim_x*j];
		}
	}

	for(int j=dim_x; j<dim_x+2; j++)
	{
		for(int i=0; i<dim_y; i++)
		{
			extended[2+i][2+j]=image[i+dim_x*(j-2)];
		}
	}

	for(int j(0); j<dim_x; j++)
	{
		for(int i(dim_y); i<dim_y+2; i++)
		{
			extended[2+i][2+j]=image[i-2+dim_x*j];
		}
	}
/*
	for(int j=0; j<dim_x+4; j++)
	{
		for(int i=0; i<dim_y+4; i++)
		{
			std::cout<<"extended["<<i+(dim_x+4)*j<<"] = "<<extended[i][j]<<std::endl;
		}
	}
*/

	kCenterY = dim_k/2+1;
	kCenterX = kCenterY;

	for(int j(1);j<=dim_x+4;j++)
	{
		for(int i(1); i<=dim_y+4; i++)
		{
			for(int m(1); m<=dim_k ; m++)
			{
				mm = dim_k - m + 1;

				for(int n(1);n<=dim_k;n++)
				{
					nn = dim_k - n + 1;

					ii = i + (m - kCenterY);
					jj = j + (n - kCenterX);

					if( ii >= 1 && ii < dim_y+4 && jj>=1 && jj< dim_x+4 )
					{
						ext_conv[i-1][j-1] += extended[ii-1][jj-1]*M.kernel[mm-1][nn-1];
					}
				}
			}
		}
	}

	for(int j(0);j<dim_x;j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			conv[i+dim_x*j] = ext_conv[2+i][2+j];
		}
	}
}


void algo_rohsa::convolution_2D_mirror(const parameters &M, const std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &conv, int dim_y, int dim_x, int dim_k)
{
	int ii(0),jj(0),mm(0),nn(0),kCenterY(0), kCenterX(0);

	std::vector <std::vector<double>> ext_conv(dim_y+4, std::vector<double>(dim_x+4,0.));
	std::vector <std::vector<double>> extended(dim_y+4, std::vector<double>(dim_x+4,0.));

	for(int j(0); j<dim_x; j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			extended[2+i][2+j]=image[i][j];
		}
	}

	for(int j(0); j<2; j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			extended[2+i][j] = image[i][j];
		}
	}

	for(int i(0); i<2; i++)
	{
		for(int j(0); j<dim_x; j++)
		{
			extended[i][2+j] = image[i][j];
		}
	}

	for(int j=dim_x; j<dim_x+2; j++)
	{
		for(int i=0; i<dim_y; i++)
		{
			extended[2+i][2+j]=image[i][j-2];
		}
	}

	for(int j(0); j<dim_x; j++)
	{
		for(int i(dim_y); i<dim_y+2; i++)
		{
			extended[2+i][2+j]=image[i-2][j];
		}
	}
	kCenterY = dim_k/2+1;
	kCenterX = kCenterY;

	for(int j(1);j<=dim_x+4;j++)
	{
		for(int i(1); i<=dim_y+4; i++)
		{
			for(int m(1); m<=dim_k ; m++)
			{
				mm = dim_k - m + 1;

				for(int n(1);n<=dim_k;n++)
				{
					nn = dim_k - n + 1;

					ii = i + (m - kCenterY);
					jj = j + (n - kCenterX);

					if( ii >= 1 && ii < dim_y+4 && jj>=1 && jj< dim_x+4 )
					{
						ext_conv[i-1][j-1] += extended[ii-1][jj-1]*M.kernel[mm-1][nn-1];
					}
				}
			}
		}
	}

	for(int j(0);j<dim_x;j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			conv[i][j] = ext_conv[2+i][2+j];
		}
	}
}
// // L'ordre x,y,lambda est celui du code fortran : lambda,y,x      pk?

// It transforms a 1D vector into a contiguous flattened 1D array from a 2D array, like a valarray

void algo_rohsa::ravel_2D(const std::vector<std::vector<double>> &map, std::vector<double> &vector, int dim_y, int dim_x)
{
	int i__=0;

	for(int k(0); k<dim_x; k++)
	{
		for(int j(0);j<dim_y;j++)
		{
			vector[i__] = map[k][j];
			i__++;
		}
	}

}

// It transforms a 1D vector into a contiguous flattened 1D array from a 3D array, the interest is close to the valarray's one

void algo_rohsa::ravel_3D_bis(const std::vector<std::vector<std::vector<double>>> &cube, double vector[], int dim_v, int dim_y, int dim_x)
{
/*
    int i__(0);
	for(int i(0); i<dim_x; i++)
		{
        for(int j(0); j<dim_y; j++)
            {
	        for(int k(0); k<dim_v; k++)
    		    {
	                vector[i__] = cube[i][j][k];
       	            i__++;
				}
            }
	    }
*/

	for(int i(0); i<dim_x; i++)
		{
        for(int j(0); j<dim_y; j++)
            {
	        for(int k(0); k<dim_v; k++)
    		    {
                vector[i*dim_y*dim_v+j*dim_v+k] = cube[i][j][k];
				}
            }
	    }


}


void algo_rohsa::ravel_3D(const std::vector<std::vector<std::vector<double>>> &cube_3D, std::vector<double> &vector, int dim_v, int dim_y, int dim_x)
{
        int i__ = 0;

/*
			for(int i(0); i<dim_v; i++)
			{
            for(int j(0); j<dim_y; j++)
                {
		        for(int k(0); k<dim_x; k++)
        {
	                        vector[i__] = cube[i][j][k];
        	                i__++;
			}
                }
        }
*/
	std::cout << "dim_v : " << dim_v <<  std::endl;
	std::cout << "dim_y : " << dim_y <<  std::endl;
	std::cout << "dim_x : " << dim_x <<  std::endl;

	std::cout << "vector.size() : " << vector.size() <<  std::endl;
	std::cout << "cube.size() : " << cube_3D.size() << " , " << cube_3D[0].size() << " , " << cube_3D[0][0].size() <<  std::endl;

//	std::cout << "avant cube[0][0][0] " <<  std::endl;


/*
    for(int k(0); k<dim_x; k++)
    {
        for(int j(0); j<dim_y; j++)
	        {
			for(int i(0); i<dim_v; i++)
			{
	            vector[i__] = cube_3D[i][j][k];
				i__++;
			}
		}
    }
*/
    for(int k(0); k<dim_x; k++)
	    {
        for(int j(0); j<dim_y; j++)
	        {
			for(int i(0); i<dim_v; i++)
				{
	            vector[k*dim_y*dim_v+j*dim_v+i] = cube_3D[i][j][k];
				}
			}
	    }
//	std::cout << "avant cube[v][y][x] " <<  std::endl;
}

void algo_rohsa::ravel_3D(const std::vector<std::vector<std::vector<double>>> &cube, double vector[], int dim_v, int dim_y, int dim_x)
{


//	code ok
/*
    int i__=0;
    for(int k(0); k<dim_x; k++)
        {
        for(int j(0); j<dim_y; j++)
            {
			for(int i(0); i<dim_v; i++)
				{
	            vector[i__] = cube[i][j][k];
        	    i__++;
				}
            }
    	}
*/

    for(int k(0); k<dim_x; k++)
        {
        for(int j(0); j<dim_y; j++)
            {
			for(int i(0); i<dim_v; i++)
				{
	            vector[k*dim_y*dim_v+j*dim_v+i] = cube[i][j][k];
				}
            }
    	}
}


void algo_rohsa::ravel_3D_abs(const std::vector<std::vector<std::vector<double>>> &cube, const std::vector<std::vector<std::vector<double>>> &cube_abs, std::vector<double> &vector, int dim_v, int dim_y, int dim_x)
{
        int i__(1);

        for(int k(0); k<dim_x; k++)
        {
                for(int j(0); j<dim_y; j++)
                {
                        for(int i(0); i<dim_v; i++)
                        {
                                vector[i__] = cube[i][j][k];
                                i__++;
                        }
                }
        }

	for(int k(0); k<dim_x; k++)
        {
                for(int j(0); j<dim_y; j++)
                {
                        for(int i(0); i<dim_v; i++)
                   	{
                                vector[i__] = cube_abs[i][j][k];
                                i__++;
                        }
                }
        }
}


// It transforms a 1D vector into a 3D array, like the step we went through when analysing data from CCfits which returns a valarray that needs to be expended into a 3D array (it's the data cube)

void algo_rohsa::unravel_3D(const std::vector<double> &vector, std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x)
{
	int k,j,i;
/*
	int i__(0);
	for(k=0; k<dim_x; k++)
	{
		for(j=0; j<dim_y; j++)
		{
			for(i=0; i<dim_v; i++)
			{
				cube[i][j][k] = vector[i__];
				i__++;
			}
		}
	}
*/
	for(k=0; k<dim_x; k++)
	{
		for(j=0; j<dim_y; j++)
		{
			for(i=0; i<dim_v; i++)
			{
				cube[i][j][k] = vector[k*dim_y*dim_v+dim_v*j+i];
			}
		}
	}

}
//°
void algo_rohsa::unravel_3D(double vector[], std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x)
{
/*
	int i__=0;
	for(int k=0; k<dim_x; k++)
	{
		for(int j=0; j<dim_y; j++)
		{
			for(int i=0; i<dim_v; i++)
			{
				cube[i][j][k] = vector[i__];
				i__++;
			}
		}
	}
*/
	for(int k=0; k<dim_x; k++)
	{
		for(int j=0; j<dim_y; j++)
		{
			for(int i=0; i<dim_v; i++)
			{
				cube[i][j][k] = vector[dim_y*dim_v*k+dim_v*j+i];
			}
		}
	}


}
//°°
// transposes i and j
// I wanted to solve cache default in dF_over_dB routine, I needed to transpose params
void algo_rohsa::unravel_3D_with_formula_transpose_xy(double vector[], std::vector<std::vector<std::vector<double>>> &cube, int dim_x, int dim_y, int dim_z)
{
int i,j,k;
#pragma omp parallel private(j,i,k) shared(cube, vector,dim_x,dim_y,dim_z)
{
#pragma omp for
	for(j=0; j<dim_y; j++)
		{
	for(i=0; i<dim_x; i++)
		{
	for(k=0; k<dim_z; k++)
	{
					cube[j][i][k] = vector[k+dim_z*j+dim_y*dim_z*i];
			}
		}
	}
}
/* //utilisé juste avant
	int i__=0;

	for(int i=0; i<dim_v; i++)
		{
	for(int j=0; j<dim_y; j++)
		{
	for(int k=0; k<dim_x; k++)
	{
					cube[i][j][k] = vector[i__];
				i__++;
			}
		}
	}
*/
}

void algo_rohsa::unravel_3D_T(double vector[], std::vector<std::vector<std::vector<double>>> &cube, int dim_x, int dim_y, int dim_z)
{
	for(int i=0; i<dim_z; i++)
		{
	for(int j=0; j<dim_y; j++)
		{
	for(int k=0; k<dim_x; k++)
	{
					cube[j][i][k] = vector[i*dim_x*dim_y+j*dim_x+k];
			}
		}
	}

/*
	int i__=0;
	for(int i=0; i<dim_z; i++)
		{
	for(int j=0; j<dim_y; j++)
		{
	for(int k=0; k<dim_x; k++)
	{
					cube[j][i][k] = vector[i__];
				i__++;
			}
		}
	}
*/

}
void algo_rohsa::unravel_3D_abs(const std::vector<double> &vector, std::vector<std::vector<std::vector<double>>> &cube_abs,std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x)
{
        int i__(0);

        for(int k(0); k<dim_x; k++)
        {
                for(int j(0); j<dim_y; j++)
                {
                        for(int i(0); i<dim_v; i++)
                        {
                                cube[i][j][k] = vector[i__];
                                i__++;
                        }
                }
        }

        for(int k(0); k<dim_x; k++)
        {
                for(int j(0); j<dim_y; j++)
                {
                        for(int i(0); i<dim_v; i++)
                        {
                                cube_abs[i][j][k] = vector[i__];
                                i__++;
                        }
                }
        }
}

// It returns the mean value of a 1D vector
double algo_rohsa::mean(const std::vector<double> &array)
{
 	return std::accumulate(array.begin(), array.end(), 0.)/std::max(1.,double(array.size()));
}

// It returns the standard deviation value of a 1D vector
// BEWARE THE STD LIBRARY 
// "Std" rather than "std"

double algo_rohsa::Std(const std::vector<double> &array)
{
	double mean_(0.), var(0.);
	int n = array.size();
	mean_ = mean(array);

	for(int i(0); i<n; i++)
	{
		var+=pow(array[i]-mean_,2);
	}
	return sqrt(var/(n-1));
}

double algo_rohsa::std_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x)
{

	std::vector<double> vector(dim_x*dim_y, 0.);
	ravel_2D(map, vector, dim_y, dim_x);
	return Std(vector);
}


double algo_rohsa::max_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x)
{
	std::vector<double> vector(dim_x*dim_y,0.);
	ravel_2D(map, vector, dim_y, dim_x);
	double val_max = vector[0];
	for (unsigned int i = 0; i < vector.size(); i++)
		if (vector[i] > val_max)
    			val_max = vector[i];
	vector.clear();
	return val_max;
}

double algo_rohsa::mean_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x)
{
	std::vector<double> vector(dim_y*dim_x, 0.);
	ravel_2D(map, vector, dim_y, dim_x);
	double mean_2D = mean(vector);
	vector.clear();
	return mean_2D;
}

void algo_rohsa::std_spectrum(int dim_x, int dim_y, int dim_v)
{
	for(int i(0); i<dim_v; i++)
	{

		std::vector<std::vector<double>> map(dim_x, std::vector<double>(dim_y,0.));

		for(int j(0); j< dim_y; j++)
		{
			for( int k(0); k<dim_x; k++)
			{
				map[k][j] = file.data[k][j][i];
			}
		}
		std_spect.vector::push_back(std_2D(map, dim_y, dim_x));
		
	}
}

void algo_rohsa::mean_spectrum(int dim_x, int dim_y, int dim_v)
{

	for(int i(0);i<dim_v;i++)
	{
		std::vector<std::vector<double>> map(dim_x, std::vector<double>(dim_y, 0.));
		for(int j(0); j<dim_y ; j++)
		{
			for(int k(0); k<dim_x ; k++)
			{
				map[k][j] = file.data[k][j][i];
			}
		}

		mean_spect.vector::push_back(mean_2D(map, dim_y, dim_x));
		map.clear();
	}
}

void algo_rohsa::max_spectrum(int dim_x, int dim_y, int dim_v)
{
	for(int i(0); i<dim_v; i++)
	{
		std::vector<std::vector<double>> map(dim_x, std::vector<double>(dim_y,0.));
		for(int j(0); j< dim_y; j++)
		{
			for( int k(0); k<dim_x; k++)
			{
				map[k][j] = file.data[k][j][i];
			}
		}
		max_spect.vector::push_back(max_2D(map, dim_y, dim_x));
		map.clear();
	}
}

void algo_rohsa::max_spectrum_norm(int dim_x, int dim_y, int dim_v, double norm_value)
{
	for(int i(0); i<dim_v; i++)
	{
		std::vector<std::vector<double>> map(dim_x, std::vector<double>(dim_y));
		for(int j(0); j< dim_y; j++)
		{
			for( int k(0); k<dim_x; k++)
			{
				map[k][j] = file.data[k][j][i];
			}
		}
		max_spect_norm.vector::push_back(max_2D(map, dim_y, dim_x));
		map.clear();
	}

	double val_max = max_spect_norm[0];
	for (unsigned int i = 0; i < max_spect_norm.size(); i++)
		if (max_spect_norm[i] > val_max)
    			val_max = max_spect_norm[i];

	for(int i(0); i<dim_v ; i++)
	{
		max_spect_norm[i] /= val_max/norm_value; 
	}
}


void algo_rohsa::mean_parameters(std::vector<std::vector<std::vector<double>>> &params)
{
	int dim1 = params.size();
	int dim2 = params[0].size();
	int dim3 = params[0][0].size();
	
	for(int p=0; p<dim1;p++){
		double mean = 0.;
		for(int i=0;i<dim2;i++){
			for(int j=0;j<dim3;j++){
				mean += params[p][j][i];
			}
		}
		mean = mean/(dim2*dim3);
		if (p%3 ==0)
			printf("Gaussienne n°%d, par n°%d, moyenne a     = %f \n", (p-p%3)/3, p, mean);
		if (p%3 ==1)
			printf("Gaussienne n°%d, par n°%d, moyenne mu    = %f \n", (p-p%3)/3, p, mean);
		if (p%3 ==2)
			printf("Gaussienne n°%d, par n°%d, moyenne sigma = %f \n", (p-p%3)/3, p, mean);
	}

}


void algo_rohsa::descente_sans_regu(parameters &M, std::vector<std::vector<std::vector<double>>> &grid_params, std::vector<std::vector<std::vector<double>>> &fit_params){
	
}
