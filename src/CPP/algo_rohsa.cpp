#include "algo_rohsa.hpp"
#include <array>
#include <chrono>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include "gradient.hpp"
#include "convolutions.hpp"
#include "f_g_cube_gpu.hpp"
#include "culbfgsb.h"
#include "callback_cpu.h"

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
		descente<double>(M, grid_params, fit_params);
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

//void algo_rohsa::descente(parameters &M, std::vector<std::vector<std::vector<double>>> &grid_params, std::vector<std::vector<std::vector<double>>> &fit_params){	
template <typename T> void algo_rohsa::descente(parameters &M, std::vector<std::vector<std::vector<T>>> &grid_params, std::vector<std::vector<std::vector<T>>> &fit_params){	
	std::vector<T> b_params(M.n_gauss,0.);
	temps_global = 0.; 
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
	this->temps_transfert_d = 0.;
	this->temps_copy = 0.;
	for(int i=0;i<M.n_gauss; i++){
		fit_params[0+3*i][0][0] = 0.;
		fit_params[1+3*i][0][0] = 1.;
		fit_params[2+3*i][0][0] = 1.;
	}
	std::cout << "fit_params.size() : " << fit_params.size() << " , " << fit_params[0].size() << " , " << fit_params[0][0].size() <<  std::endl;

	std::cout << "grid_params.size() : "<< grid_params.size() << " , " << grid_params[0].size()  << " , " << grid_params[0][0].size() << std::endl;

	double temps2_before_nside;
	double temps1_descente = omp_get_wtime();

	std::vector<T> fit_params_flat(fit_params.size(),0.); //used below

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

	if(!(M.jump_to_last_level)){
		for(n=0; n<file.nside; n++)
		{
			double temps1_init_spectrum = omp_get_wtime();

			int power(pow(2,n));

			std::cout << " power = " << power << std::endl;

			std::vector<std::vector<std::vector<T>>> cube_mean(power, std::vector<std::vector<T>>(power,std::vector<T>(dim_v,1.)));

			mean_array(power, cube_mean);

			std::vector<T> cube_mean_flat(cube_mean[0][0].size());

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
	
	//	(this->file).plot_multi_lines(fit_params, cube_mean);

	/*
		for(int i(0); i<3*M.n_gauss; i++) {
			for(int j(0); j<power; j++) {
				for(int k(0); k<power; k++) {
					printf("fit_params[%d][%d][%d]=%f\n", i,j,k, fit_params[i][j][k]);
				}
			}
		}
	//exit(0);
	*/
						double temps1_upgrade = omp_get_wtime();
						upgrade(M ,cube_mean, fit_params, power);
						double temps2_upgrade = omp_get_wtime();
						temps_upgrade+=temps2_upgrade-temps1_upgrade;

	//	(this->file).plot_multi_lines(fit_params, cube_mean, "second_plot");

	/*
		for(int i(0); i<3*M.n_gauss; i++) {
			for(int j(0); j<power; j++) {
				for(int k(0); k<power; k++) {
					printf("fit_params[%d][%d][%d]=%f\n", i,j,k, fit_params[i][j][k]);
				}
			}
		}
	exit(0);
	*/
					}
					if (n>0 and n<file.nside){

						if (power ==2){
/*
			fit_params[           0 ][           0 ][           0 ] =   0.12800975412547183      ;
			fit_params[           0 ][           0 ][           1 ] =   0.12800975412547183      ;
			fit_params[           0 ][           1 ][           0 ] =   0.12800975412547183      ;
			fit_params[           0 ][           1 ][           1 ] =   0.12800975412547183      ;
			fit_params[           1 ][           0 ][           0 ] =    180.05482716213552      ;
			fit_params[           1 ][           0 ][           1 ] =    180.05482716213552      ;
			fit_params[           1 ][           1 ][           0 ] =    180.05482716213552      ;
			fit_params[           1 ][           1 ][           1 ] =    180.05482716213552      ;
			fit_params[           2 ][           0 ][           0 ] =    12.106686744104559      ;
			fit_params[           2 ][           0 ][           1 ] =    12.106686744104559      ;
			fit_params[           2 ][           1 ][           0 ] =    12.106686744104559      ;
			fit_params[           2 ][           1 ][           1 ] =    12.106686744104559      ;
			fit_params[           3 ][           0 ][           0 ] =    1.0600407207659746      ;
			fit_params[           3 ][           0 ][           1 ] =    1.0600407207659746      ;
			fit_params[           3 ][           1 ][           0 ] =    1.0600407207659746      ;
			fit_params[           3 ][           1 ][           1 ] =    1.0600407207659746      ;
			fit_params[           4 ][           0 ][           0 ] =    250.52298969665108      ;
			fit_params[           4 ][           0 ][           1 ] =    250.52298969665108      ;
			fit_params[           4 ][           1 ][           0 ] =    250.52298969665108      ;
			fit_params[           4 ][           1 ][           1 ] =    250.52298969665108      ;
			fit_params[           5 ][           0 ][           0 ] =    14.238167675192740      ;
			fit_params[           5 ][           0 ][           1 ] =    14.238167675192740      ;
			fit_params[           5 ][           1 ][           0 ] =    14.238167675192740      ;
			fit_params[           5 ][           1 ][           1 ] =    14.238167675192740      ;
			fit_params[           6 ][           0 ][           0 ] =    1.2084617713378702      ;
			fit_params[           6 ][           0 ][           1 ] =    1.2084617713378702      ;
			fit_params[           6 ][           1 ][           0 ] =    1.2084617713378702      ;
			fit_params[           6 ][           1 ][           1 ] =    1.2084617713378702      ;
			fit_params[           7 ][           0 ][           0 ] =    224.48625973277666      ;
			fit_params[           7 ][           0 ][           1 ] =    224.48625973277666      ;
			fit_params[           7 ][           1 ][           0 ] =    224.48625973277666      ;
			fit_params[           7 ][           1 ][           1 ] =    224.48625973277666      ;
			fit_params[           8 ][           0 ][           0 ] =    11.357933194038875      ;
			fit_params[           8 ][           0 ][           1 ] =    11.357933194038875      ;
			fit_params[           8 ][           1 ][           0 ] =    11.357933194038875      ;
			fit_params[           8 ][           1 ][           1 ] =    11.357933194038875      ;
			fit_params[           9 ][           0 ][           0 ] =    2.3550818551629462e-2 ;
			fit_params[           9 ][           0 ][           1 ] =    2.3550818551629462e-2 ;
			fit_params[           9 ][           1 ][           0 ] =    2.3550818551629462e-2 ;
			fit_params[           9 ][           1 ][           1 ] =    2.3550818551629462e-2 ;
			fit_params[          10 ][           0 ][           0 ] =    142.67171188631278      ;
			fit_params[          10 ][           0 ][           1 ] =    142.67171188631278      ;
			fit_params[          10 ][           1 ][           0 ] =    142.67171188631278      ;
			fit_params[          10 ][           1 ][           1 ] =    142.67171188631278      ;
			fit_params[          11 ][           0 ][           0 ] =    24.303045603972546      ;
			fit_params[          11 ][           0 ][           1 ] =    24.303045603972546      ;
			fit_params[          11 ][           1 ][           0 ] =    24.303045603972546      ;
			fit_params[          11 ][           1 ][           1 ] =    24.303045603972546      ;
			fit_params[          12 ][           0 ][           0 ] =    2.4168148793963518      ;
			fit_params[          12 ][           0 ][           1 ] =    2.4168148793963518      ;
			fit_params[          12 ][           1 ][           0 ] =    2.4168148793963518      ;
			fit_params[          12 ][           1 ][           1 ] =    2.4168148793963518      ;
			fit_params[          13 ][           0 ][           0 ] =    203.40124971351946      ;
			fit_params[          13 ][           0 ][           1 ] =    203.40124971351946      ;
			fit_params[          13 ][           1 ][           0 ] =    203.40124971351946      ;
			fit_params[          13 ][           1 ][           1 ] =    203.40124971351946      ;
			fit_params[          14 ][           0 ][           0 ] =    11.067689564722622      ;
			fit_params[          14 ][           0 ][           1 ] =    11.067689564722622      ;
			fit_params[          14 ][           1 ][           0 ] =    11.067689564722622      ;
			fit_params[          14 ][           1 ][           1 ] =    11.067689564722622      ;
			fit_params[          15 ][           0 ][           0 ] =    6.5103157015035176e-2 ;
			fit_params[          15 ][           0 ][           1 ] =    6.5103157015035176e-2 ;
			fit_params[          15 ][           1 ][           0 ] =    6.5103157015035176e-2 ;
			fit_params[          15 ][           1 ][           1 ] =    6.5103157015035176e-2;
			fit_params[          16 ][           0 ][           0 ] =    272.23544007289087      ;
			fit_params[          16 ][           0 ][           1 ] =    272.23544007289087      ;
			fit_params[          16 ][           1 ][           0 ] =    272.23544007289087      ;
			fit_params[          16 ][           1 ][           1 ] =    272.23544007289087      ;
			fit_params[          17 ][           0 ][           0 ] =    14.676429309765949      ;
			fit_params[          17 ][           0 ][           1 ] =    14.676429309765949      ;
			fit_params[          17 ][           1 ][           0 ] =    14.676429309765949      ;
			fit_params[          17 ][           1 ][           1 ] =    14.676429309765949      ;
			fit_params[          18 ][           0 ][           0 ] =   0.55919737225271182      ;
			fit_params[          18 ][           0 ][           1 ] =   0.55919737225271182      ;
			fit_params[          18 ][           1 ][           0 ] =   0.55919737225271182      ;
			fit_params[          18 ][           1 ][           1 ] =   0.55919737225271182      ;
			fit_params[          19 ][           0 ][           0 ] =    197.19727306460263      ;
			fit_params[          19 ][           0 ][           1 ] =    197.19727306460263      ;
			fit_params[          19 ][           1 ][           0 ] =    197.19727306460263      ;
			fit_params[          19 ][           1 ][           1 ] =    197.19727306460263      ;
			fit_params[          20 ][           0 ][           0 ] =    3.7498051053837402      ;
			fit_params[          20 ][           0 ][           1 ] =    3.7498051053837402      ;
			fit_params[          20 ][           1 ][           0 ] =    3.7498051053837402      ;
			fit_params[          20 ][           1 ][           1 ] =    3.7498051053837402      ;
			fit_params[          21 ][           0 ][           0 ] =   0.37501802803217477      ;
			fit_params[          21 ][           0 ][           1 ] =   0.37501802803217477      ;
			fit_params[          21 ][           1 ][           0 ] =   0.37501802803217477      ;
			fit_params[          21 ][           1 ][           1 ] =   0.37501802803217477      ;
			fit_params[          22 ][           0 ][           0 ] =    203.32039391313074      ;
			fit_params[          22 ][           0 ][           1 ] =    203.32039391313074      ;
			fit_params[          22 ][           1 ][           0 ] =    203.32039391313074      ;
			fit_params[          22 ][           1 ][           1 ] =    203.32039391313074      ;
			fit_params[          23 ][           0 ][           0 ] =    2.7070692040635058      ;
			fit_params[          23 ][           0 ][           1 ] =    2.7070692040635058      ;
			fit_params[          23 ][           1 ][           0 ] =    2.7070692040635058      ;
			fit_params[          23 ][           1 ][           1 ] =    2.7070692040635058      ;
			fit_params[          24 ][           0 ][           0 ] =   0.51704192683721151      ;
			fit_params[          24 ][           0 ][           1 ] =   0.51704192683721151      ;
			fit_params[          24 ][           1 ][           0 ] =   0.51704192683721151      ;
			fit_params[          24 ][           1 ][           1 ] =   0.51704192683721151      ;
			fit_params[          25 ][           0 ][           0 ] =    200.07319126869734      ;
			fit_params[          25 ][           0 ][           1 ] =    200.07319126869734      ;
			fit_params[          25 ][           1 ][           0 ] =    200.07319126869734      ;
			fit_params[          25 ][           1 ][           1 ] =    200.07319126869734      ;
			fit_params[          26 ][           0 ][           0 ] =    12.457897408187174      ;
			fit_params[          26 ][           0 ][           1 ] =    12.457897408187174      ;
			fit_params[          26 ][           1 ][           0 ] =    12.457897408187174      ;
			fit_params[          26 ][           1 ][           1 ] =    12.457897408187174      ;
			fit_params[          27 ][           0 ][           0 ] =    5.0955018710702592e-2 ;
			fit_params[          27 ][           0 ][           1 ] =    5.0955018710702592e-2;
			fit_params[          27 ][           1 ][           0 ] =    5.0955018710702592e-2;
			fit_params[          27 ][           1 ][           1 ] =    5.0955018710702592e-2 ;
			fit_params[          28 ][           0 ][           0 ] =    297.76970529211712      ;
			fit_params[          28 ][           0 ][           1 ] =    297.76970529211712      ;
			fit_params[          28 ][           1 ][           0 ] =    297.76970529211712      ;
			fit_params[          28 ][           1 ][           1 ] =    297.76970529211712      ;
			fit_params[          29 ][           0 ][           0 ] =    12.161470649157568      ;
			fit_params[          29 ][           0 ][           1 ] =    12.161470649157568      ;
			fit_params[          29 ][           1 ][           0 ] =    12.161470649157568      ;
			fit_params[          29 ][           1 ][           1 ] =    12.161470649157568      ;
			fit_params[          30 ][           0 ][           0 ] =    6.8536627841306402e-2 ;
			fit_params[          30 ][           0 ][           1 ] =    6.8536627841306402e-2 ;
			fit_params[          30 ][           1 ][           0 ] =    6.8536627841306402e-2 ;
			fit_params[          30 ][           1 ][           1 ] =    6.8536627841306402e-2 ;
			fit_params[          31 ][           0 ][           0 ] =    257.58792869697089      ;
			fit_params[          31 ][           0 ][           1 ] =    257.58792869697089      ;
			fit_params[          31 ][           1 ][           0 ] =    257.58792869697089      ;
			fit_params[          31 ][           1 ][           1 ] =    257.58792869697089      ;
			fit_params[          32 ][           0 ][           0 ] =    3.2948315664618648      ;
			fit_params[          32 ][           0 ][           1 ] =    3.2948315664618648      ;
			fit_params[          32 ][           1 ][           0 ] =    3.2948315664618648      ;
			fit_params[          32 ][           1 ][           1 ] =    3.2948315664618648      ;
			fit_params[          33 ][           0 ][           0 ] =    4.2440241626722545e-2 ;
			fit_params[          33 ][           0 ][           1 ] =    4.2440241626722545e-2 ;
			fit_params[          33 ][           1 ][           0 ] =    4.2440241626722545e-2 ;
			fit_params[          33 ][           1 ][           1 ] =    4.2440241626722545e-2 ;
			fit_params[          34 ][           0 ][           0 ] =    214.96764833274480      ;
			fit_params[          34 ][           0 ][           1 ] =    214.96764833274480      ;
			fit_params[          34 ][           1 ][           0 ] =    214.96764833274480      ;
			fit_params[          34 ][           1 ][           1 ] =    214.96764833274480      ;
			fit_params[          35 ][           0 ][           0 ] =    2.2912606146642367      ;
			fit_params[          35 ][           0 ][           1 ] =    2.2912606146642367      ;
			fit_params[          35 ][           1 ][           0 ] =    2.2912606146642367      ;
			fit_params[          35 ][           1 ][           1 ] =    2.2912606146642367      ;
			b_params[           0 ] =    12.000000000000000      ;
			b_params[           1 ] =    12.000000000000000      ;
			b_params[           2 ] =    12.000000000000000      ;
			b_params[           3 ] =    12.000000000000000      ;
			b_params[           4 ] =    11.756541084302166      ;
			b_params[           5 ] =    11.892606675545769      ;
			b_params[           6 ] =    3.8724543060279322      ;
			b_params[           7 ] =    2.7406645675468209      ;
			b_params[           8 ] =    11.523657275718318      ;
			b_params[           9 ] =    10.349266870430457      ;
			b_params[          10 ] =    3.1369301998655779      ;
			b_params[          11 ] =    1.8009704780789586      ;
*/

						}
						std::vector<std::vector<T>> std_map(power, std::vector<T>(power,0.));
						double temps_std_map1=omp_get_wtime();
						if (M.noise){
							//reshape_noise_up(indice_debut, indice_fin);
							//mean_map()	
						}
						
						else if (M.noise==false){
							set_stdmap_transpose<T>(std_map, cube_mean, M.lstd, M.ustd);

	//						set_stdmap(std_map, cube_mean, M.lstd, M.ustd); //?
						}
						double temps_std_map2=omp_get_wtime();
						temps_std_map_pp+=temps_std_map2-temps_std_map1;

						double temps1_update_pp=omp_get_wtime();

						update_clean<T>(M, cube_mean, fit_params, std_map, power, power, dim_v, b_params);

	//					(this->file).plot_multi_lines(fit_params, cube_mean, std::to_string(power));

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
		temps2_before_nside = omp_get_wtime();

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
		std::cout<< "	-> Temps de transfert : " << this->temps_transfert_d <<std::endl;
		std::cout<< "temps d'exécution dF/dB : "<<temps_f_g_cube<<std::endl;
		std::cout<< "Temps d'exécution convolution : " << temps_conv <<std::endl;
		std::cout<< "Temps d'exécution deriv : " << temps_deriv  <<std::endl;
		std::cout<< "Temps d'exécution ravel_3D : " << temps_ravel <<std::endl;
		std::cout<< "Temps d'exécution tableaux : " << temps_tableaux <<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"           fin détails             "<<std::endl;
		std::cout<<"                                "<<std::endl;




		//nouvelle place de reshape_down
		int offset_w = (this->dim_cube[0]-this->dim_data[0])/2;
		int offset_h = (this->dim_cube[1]-this->dim_data[1])/2;

		std::cout<<"Taille fit_params : "<<fit_params.size()<<" , "<<fit_params[0].size()<<" , "<<fit_params[0][0].size()<<std::endl;
		std::cout<<"Taille grid_params : "<<grid_params.size()<<" , "<<grid_params[0].size()<<" , "<<grid_params[0][0].size()<<std::endl;

		//ancienne place de reshape_down	
		double temps_reshape_down2 = omp_get_wtime();
		reshape_down<T>(fit_params, grid_params);
		double temps_reshape_down1 = omp_get_wtime();
		temps_reshape_down2 += temps_reshape_down2-temps_reshape_down1;

		std::cout<<"Après reshape_down"<<std::endl;
		if(M.save_second_to_last_level){
			this->file.write_in_file(grid_params);
		}

	}else{
		this->file.get_from_file(grid_params, 3*M.n_gauss, this->dim_data[1], this->dim_data[0]);
	}

	this->grid_params = grid_params;
	std::vector<std::vector<T>> std_map(this->dim_data[1], std::vector<T>(this->dim_data[0],0.));
	double temps_dernier_niveau1 = omp_get_wtime();

	double temps_std_map1=omp_get_wtime();
	if(M.noise){
		//std_map=std_data;
	} else {

		set_stdmap<T>(std_map, this->file.data, M.lstd, M.ustd);
	}
	double temps_std_map2=omp_get_wtime();
	temps_std_map_dp+=temps_std_map2-temps_std_map1;

	double temps_update_dp1 = omp_get_wtime();

	if(M.regul){
		std::cout<<"Updating last level"<<std::endl;
		// repère recherche pb %
		update_clean(M, this->file.data, grid_params, std_map, this->dim_data[0], this->dim_data[1], this->dim_v, b_params);
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
/*	std::cout<<"temps d'exécution résidu et f : "<<temps_f_g_cube<<std::endl;
	std::cout<<"temps d'exécution setulb : "<<temps_setulb<<std::endl;
	std::cout<< "Temps d'exécution convolution : " << temps_conv <<std::endl;
	std::cout<< "	-> Temps d'exécution routine mirroir : " << temps_mirroirs <<std::endl;
	std::cout<< "	-> Temps de transfert : " << this->temps_transfert_d + temps_transfert <<std::endl;
	std::cout<< "Temps d'exécution deriv : " << temps_deriv  <<std::endl;
	std::cout<< "Temps d'exécution ravel_3D : " << temps_ravel <<std::endl;
	std::cout<< "Temps d'exécution tableaux avant gradient : " << temps_tableaux <<std::endl;
	std::cout<< "Temps d'exécution tableaux dans update : " << temps_tableau_update <<std::endl;
*/
	std::cout<<"Temps d'exécution setulb : "<<temps_setulb<<std::endl;
	std::cout<<"Temps d'exécution f_g_cube : "<<this->temps_f_g_cube<<std::endl;
	std::cout<< "	-> Temps d'exécution transfert données : " << this->temps_copy  <<std::endl;
	std::cout<< "	-> Temps d'exécution attache aux données : " << this->temps_tableaux <<std::endl;
	std::cout<< "	-> Temps d'exécution deriv : " << this->temps_deriv  <<std::endl;
	std::cout<< "	-> Temps d'exécution régularisation : " << this->temps_conv <<std::endl;
	std::cout<<"                                "<<std::endl;
	std::cout<<"           fin détails             "<<std::endl;
	std::cout<<"                                "<<std::endl;
}

/*
	printf("this->temps_copy = %f\n", this->temps_copy/1000);
	printf("this->temps_tableaux = %f\n", this->temps_tableaux/1000);
	printf("this->temps_deriv = %f\n", this->temps_deriv/1000);
	printf("this->temps_conv = %f\n", this->temps_conv/1000);
	printf("this->temps_f_g_cube = %f\n", this->temps_f_g_cube/1000);

*/


template <typename T> void algo_rohsa::reshape_down(std::vector<std::vector<std::vector<T>>> &tab1, std::vector<std::vector<std::vector<T>>>&tab2)
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


template <typename T> void algo_rohsa::update_clean(parameters &M, std::vector<std::vector<std::vector<T>>> &cube_avgd_or_data, std::vector<std::vector<std::vector<T>>> &params, std::vector<std::vector<T>> &std_map, int indice_x, int indice_y, int indice_v, std::vector<T> &b_params) {

	std::cout << "params.size() : " << params.size() << " , " << params[0].size() << " , " << params[0][0].size() <<  std::endl;


//printf("b_params[0] = %f \n",b_params[0] );
//printf("beta \n");

	//cube flattened for array operations in f_g_cube_cuda
	//WARNING : free(cube_flattened) at the end of the function
	T* cube_flattened = NULL;
	size_t size_cube = indice_x*indice_y*indice_v*sizeof(T); 
	cube_flattened = (T*)malloc(size_cube);

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

	int n_beta = (3*M.n_gauss * indice_y * indice_x) +M.n_gauss;

	T* lb = NULL;
	T* ub = NULL;
	T* beta = NULL;
	size_t size_n_beta = n_beta*sizeof(T); 
	lb = (T*)malloc(size_n_beta);
	ub = (T*)malloc(size_n_beta);
	beta = (T*)malloc(size_n_beta);
	initialize_array(lb, n_beta, 0.);
	initialize_array(ub, n_beta, 0.);
	initialize_array(beta, n_beta, 0.);

	std::vector<std::vector<std::vector<T>>> lb_3D(3*M.n_gauss, std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::cout << "lb_3D.size() : " << lb_3D.size() << " , " << lb_3D[0].size() << " , " << lb_3D[0][0].size() <<  std::endl;

	std::vector<std::vector<T>> image_amp(indice_y, std::vector<T>(indice_x,0.));
	std::vector<std::vector<T>> image_mu(indice_y, std::vector<T>(indice_x,0.));
	std::vector<std::vector<T>> image_sig(indice_y, std::vector<T>(indice_x,0.));

	std::vector<T> mean_amp(M.n_gauss,0.);
	std::vector<T> mean_mu(M.n_gauss,0.);
	std::vector<T> mean_sig(M.n_gauss,0.);

	std::vector<T> ravel_amp(indice_y*indice_x,0.);
	std::vector<T> ravel_mu(indice_y*indice_x,0.);
	std::vector<T> ravel_sig(indice_y*indice_x,0.);

	std::vector<T> cube_flat(cube_avgd_or_data[0][0].size(),0.);
	std::vector<T> lb_3D_flat(lb_3D.size(),0.);

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
	exit(0);cube_flat
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

	//CLEAN
	//three_D_to_one_D(threeD, oneD, x,y,z) transforms a 3D array of dimensions z,y,x into a 1D array of dimensions x,y,z
	//lb 1difié est du format x, y, 3*ng
	three_D_to_one_D_same_dimensions(lb_3D, lb, 3*M.n_gauss, indice_y, indice_x);
	//ravel_3D(lb_3D, lb, 3*M.n_gauss, indice_y, indice_x);//ancien
	three_D_to_one_D_same_dimensions(ub_3D, ub, 3*M.n_gauss, indice_y, indice_x);
	//ravel_3D(ub_3D, ub, 3*M.n_gauss, indice_y, indice_x); //ancien
	three_D_to_one_D_same_dimensions(params, beta, 3*M.n_gauss, indice_y, indice_x);

	for(int i=0; i<M.n_gauss; i++){
		lb[n_beta-M.n_gauss+i] = M.lb_sig;
		ub[n_beta-M.n_gauss+i] = M.ub_sig;
		beta[n_beta-M.n_gauss+i] = b_params[i];
	}

	temps_tableau_update += omp_get_wtime() - temps1_tableau_update;


//	minimize_clean_gpu<T>(M, n_beta, M.m, beta, lb, ub, cube_avgd_or_data, std_map, indice_x, indice_y, indice_v, cube_flattened); 

	minimize_clean<T>(M, n_beta, M.m, beta, lb, ub, cube_avgd_or_data, std_map, indice_x, indice_y, indice_v, cube_flattened); 

//	minimize_clean_cpu<T>(M, n_beta, M.m, beta, lb, ub, cube_avgd_or_data, std_map, indice_x, indice_y, indice_v, cube_flattened); 

/*
	if(indice_x<=64){
		minimize_clean<T>(M, n_beta, M.m, beta, lb, ub, cube_avgd_or_data, std_map, indice_x, indice_y, indice_v, cube_flattened); 
	//	minimize_clean_cpu<T>(M, n_beta, M.m, beta, lb, ub, cube_avgd_or_data, std_map, indice_x, indice_y, indice_v, cube_flattened); 
	} else{
		minimize_clean_gpu<T>(M, n_beta, M.m, beta, lb, ub, cube_avgd_or_data, std_map, indice_x, indice_y, indice_v, cube_flattened); 
	}
*/

	double temps2_tableau_update = omp_get_wtime();
	//x,y,z->x,y,z
	one_D_to_three_D_same_dimensions(beta, params, 3*M.n_gauss, indice_y, indice_x);
//	unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

	int number_plot_2D = ceil(log(indice_x)/log(2));
	this->file.simple_plot_through_regu(params, 0,0,number_plot_2D);
	this->file.simple_plot_through_regu(params, 0,1,number_plot_2D);
	this->file.simple_plot_through_regu(params, 0,2,number_plot_2D);

	for(int i=0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

	temps_tableau_update += omp_get_wtime() - temps2_tableau_update;

	free(lb);
	free(ub);
	free(beta);
	free(cube_flattened);
}

template <typename T> void algo_rohsa::set_stdmap(std::vector<std::vector<T>> &std_map, std::vector<std::vector<std::vector<T>>> &cube_or_data, int lb, int ub){
	std::vector<T> line(ub-lb+1,0.);
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
template <typename T> void algo_rohsa::set_stdmap_transpose(std::vector<std::vector<T>> &std_map, std::vector<std::vector<std::vector<T>>> &cube, int lb, int ub){
	std::vector<T> line(ub-lb+1,0.);
	int dim[3];
	dim[2]=cube[0][0].size();
	dim[1]=cube[0].size();
	dim[0]=cube.size();
	for(int j=0; j<dim[1]; j++){
		for(int i=0; i<dim[0]; i++){
			for(int p=0; p<= ub-lb; p++){
				line[p] = cube[i][j][p+lb];
			}
			std_map[j][i] = Std(line);
//		printf("Std(line) = %f \n",Std(line));
		}
	}
}

template <typename T> void algo_rohsa::f_g_cube_fast(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map){
	std::vector<std::vector<std::vector<T>>> deriv(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<std::vector<std::vector<T>>> g_3D(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<std::vector<std::vector<T>>> residual(indice_x,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_v,0.)));
	std::vector<std::vector<std::vector<T>>> params(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<T> b_params(M.n_gauss,0.);

	std::vector<std::vector<T>> conv_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_sig(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_sig(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_sig(indice_y,std::vector<T>(indice_x, 0.));

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
			std::vector<T> residual_1D(indice_v,0.);
			std::vector<T> params_flat(params.size(),0.);
			std::vector<T> cube_flat(cube[0][0].size(),0.);

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
		T v_1=0.,v_2=0.,v_3=0.;

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


template <typename T> void algo_rohsa::f_g_cube_fast_clean(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map){
	std::vector<std::vector<std::vector<T>>> deriv(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<std::vector<std::vector<T>>> residual(indice_x,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_v,0.)));
	std::vector<std::vector<std::vector<T>>> params(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<T> b_params(M.n_gauss,0.);
	std::vector<std::vector<T>> conv_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_sig(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_sig(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_sig(indice_y,std::vector<T>(indice_x, 0.));

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;

	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

	for(int i = 0; i< n_beta; i++){
		g[i]=0.;
	}
	f=0.;

	double temps1_ravel = omp_get_wtime();
	one_D_to_three_D_same_dimensions(beta, params, 3*M.n_gauss, indice_y, indice_x);
	double temps2_ravel = omp_get_wtime();
	temps_ravel+=temps2_ravel-temps1_ravel;
	//unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

	double temps1_tableaux = omp_get_wtime();
	for(int j=0; j<indice_x; j++){
		for(int i=0; i<indice_y; i++){
			std::vector<T> residual_1D(indice_v,0.);
			std::vector<T> params_flat(params.size(),0.);
			std::vector<T> cube_flat(cube[0][0].size(),0.);

			for (int p=0; p<3*M.n_gauss; p++){
				params_flat[p]=params[p][i][j];
			}
			for (int p=0; p<indice_v; p++){
				cube_flat[p]=cube[j][i][p];
			}

			myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);

			for (int p=0;p<indice_v;p++){
				residual[j][i][p]=residual_1D[p];
			}

			if(std_map[i][j]>0.){
				f += myfunc_spec(residual_1D)*1/pow(std_map[i][j],2.); //std_map est arrondie... 
			}
		}
	}
	double temps2_tableaux = omp_get_wtime();

	double temps1_deriv = omp_get_wtime();
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

/*
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){
				printf("conv_conv_amp[%d][%d] = %f\n", p,q,conv_conv_amp[p][q]);
				printf("conv_conv_mu[%d][%d] = %f\n", p,q,conv_conv_mu[p][q]);
				printf("conv_conv_sig[%d][%d] = %f\n", p,q,conv_conv_sig[p][q]);
			}
		}
*/

		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){
				f+= 0.5*M.lambda_amp*pow(conv_amp[j][l],2);
				f+= 0.5*M.lambda_mu*pow(conv_mu[j][l],2);
				f+= 0.5*M.lambda_sig*pow(conv_sig[j][l],2) + 0.5*M.lambda_var_sig*pow(image_sig[j][l]-b_params[i],2);
//				printf("b_params[i] = %f\n", b_params[i]);

				g[n_beta-M.n_gauss+i] += M.lambda_var_sig*(b_params[i]-image_sig[j][l]);
				for(int k=0; k<indice_v; k++){
					if(std_map[l][j]>0.){
						T spec = T(k+1);
						deriv[0+3*i][j][l] += exp(-pow( spec-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*residual[l][j][k]/pow(std_map[j][l],2);
						deriv[1+3*i][j][l] += params[3*i][j][l]*( spec - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( spec-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*residual[l][j][k]/pow(std_map[j][l],2);
						deriv[2+3*i][j][l] += params[3*i][j][l]*pow( spec - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( spec-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*residual[l][j][k]/pow(std_map[j][l],2);
					}
				}
				double temps2_deriv = omp_get_wtime();
				temps_deriv+= temps2_deriv - temps1_deriv;
				deriv[0+3*i][j][l] += M.lambda_amp*conv_conv_amp[j][l];
				deriv[1+3*i][j][l] += M.lambda_mu*conv_conv_mu[j][l];
				deriv[2+3*i][j][l] += M.lambda_sig*conv_conv_sig[j][l]+M.lambda_var_sig*(image_sig[j][l]-b_params[i]);
			}
		}
		
	}
	double temps2_deriv = omp_get_wtime();

	temps_deriv+= temps2_deriv - temps1_deriv;

	temps_ravel+=temps2_ravel-temps1_ravel;
	temps1_ravel = omp_get_wtime();
	three_D_to_one_D_same_dimensions(deriv, g, 3*M.n_gauss, indice_y, indice_x);
	temps2_ravel = omp_get_wtime();
	temps_ravel+=temps2_ravel-temps1_ravel;
	}



template <typename T> void algo_rohsa::f_g_cube_fast_clean_optim_CPU(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map){
	bool print = false;

	std::vector<std::vector<std::vector<T>>> deriv(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<std::vector<std::vector<T>>> residual(indice_v,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<std::vector<std::vector<T>>> params(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<T> b_params(M.n_gauss,0.);
	std::vector<std::vector<T>> conv_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_sig(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_sig(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_sig(indice_y,std::vector<T>(indice_x, 0.));

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;


	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

	for(int i = 0; i< n_beta; i++){
		g[i]=0.;
	}
	f=0.;

	double temps1_ravel = omp_get_wtime();
	one_D_to_three_D_same_dimensions(beta, params, 3*M.n_gauss, indice_y, indice_x);
	double temps2_ravel = omp_get_wtime();
	temps_ravel+=temps2_ravel-temps1_ravel;
	//unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

double temps1_tableaux = omp_get_wtime();
	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			std::vector<T> residual_1D(indice_v,0.);
			std::vector<T> params_flat(params.size(),0.);
			std::vector<T> cube_flat(cube[0][0].size(),0.);

			for (int p=0; p<3*M.n_gauss; p++){
				params_flat[p]=params[p][i][j];
			}
			for (int p=0; p<indice_v; p++){
				cube_flat[p]=cube[j][i][p];
			}

			myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);

			for (int p=0;p<indice_v;p++){
				residual[p][i][j]=residual_1D[p];
			}

			if(std_map[i][j]>0.){
				f += myfunc_spec(residual_1D)*1/pow(std_map[i][j],2.); //std_map est arrondie... 
			}

		}
	}
double temps2_tableaux = omp_get_wtime();

/*
	for(int i=0; i<n_beta-M.n_gauss; i++){
		printf("beta[%d] = %f\n",i, beta[i]);
	}
	for(int k=0; k<indice_v; k++){
		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){

	printf("residual[%d][%d][%d] = %.15f\n",k,j,l, residual[k][j][l]);

			}
		}
	}
*/
if(print){
printf("--> mi-chemin : f = %.16f\n", f);
	}

double temps1_dF_dB = omp_get_wtime();
double temps2_dF_dB = omp_get_wtime();

double temps1_deriv = omp_get_wtime();
	int i;

	#pragma omp parallel private(i) shared(params,deriv,std_map,residual,indice_v,indice_y,indice_x)
	{
	#pragma omp for
	for(i=0; i<M.n_gauss; i++){
		for(int k=0; k<indice_v; k++){
			for(int j=0; j<indice_y; j++){
				for(int l=0; l<indice_x; l++){
					if(std_map[j][l]>0.){
						T spec = T(k+1);
						deriv[0+3*i][j][l] += exp(-pow( spec-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*residual[k][j][l]/pow(std_map[j][l],2);
						deriv[1+3*i][j][l] += params[3*i][j][l]*( spec - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( spec-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*residual[k][j][l]/pow(std_map[j][l],2);
						deriv[2+3*i][j][l] += params[3*i][j][l]*pow( spec - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( spec-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*residual[k][j][l]/pow(std_map[j][l],2);
					}
				}
			}
		}
	}
	}
	double temps2_deriv = omp_get_wtime();
if(print){
	for(int i=0; i<indice_v; i++){
		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){
	printf("residual[%d][%d][%d] = %.16f\n",i,j,l, residual[i][j][l]);
			}
		}
	}
	for(int k=0; k<3*M.n_gauss; k++){
		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){

	printf("deriv[%d][%d][%d] = %.16f\n",k,j,l, deriv[k][j][l]);

			}
		}
	}
	for(int j=0; j<indice_y; j++){
		for(int l=0; l<indice_x; l++){
	printf("std_map[%d][%d] = %.16f\n",j,l, std_map[j][l]);
		}
	}
}
/*
	printf("deriv[0][0][0] = %f\n", deriv[0][0][0]);
	printf("deriv[0][0][1] = %f\n", deriv[0][0][1]);
	printf("deriv[0][0][2] = %f\n", deriv[0][0][2]);
*/
	double temps1_conv = omp_get_wtime();
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

		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){
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
	double temps2_conv = omp_get_wtime();
	temps_tableaux += temps2_tableaux - temps1_tableaux;


	temps_ravel+=temps2_ravel-temps1_ravel;
	temps1_ravel = omp_get_wtime();
	three_D_to_one_D_same_dimensions(deriv, g, 3*M.n_gauss, indice_y, indice_x);
	temps2_ravel = omp_get_wtime();
	temps_ravel+=temps2_ravel-temps1_ravel;

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;

	temps_f_g_cube += temps2_dF_dB - temps1_dF_dB;

	//ravel_3D(deriv, g, 3*M.n_gauss, indice_y, indice_x);
/*
	for(int i=0; i<3*M.n_gauss; i++){
		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){
				printf("deriv[%d][%d][%d] = %f\n",i,j,l, deriv[i][j][l]);//deriv[0+3*i][j][l]
			}
		}
	}
	for(int j=0; j<n_beta; j++){
		printf("g[%d] = %f\n",j, g[j]);
	}
	exit(0);
*/
if(print){
	printf("--> fin-chemin : f = %.16f\n", f);
	std::cin.ignore();
	}
}


template <typename T> void algo_rohsa::f_g_cube_fast_clean_optim_CPU_lib(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T** assist_buffer){
	std::vector<std::vector<std::vector<T>>> deriv(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<std::vector<std::vector<T>>> residual(indice_v,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<std::vector<std::vector<T>>> params(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<T> b_params(M.n_gauss,0.);
	std::vector<std::vector<T>> conv_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_sig(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_sig(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_sig(indice_y,std::vector<T>(indice_x, 0.));

	int n_beta = n;//(3*M.n_gauss*indice_x*indice_y)+M.n_gauss;

	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

	for(int i = 0; i< n_beta; i++){
		g[i]=0.;
	}
	f=0.;

	for(int i = 0; i<n_beta; i++){
		printf("beta[%d] = %f\n",i,beta[i]);
	}

	double temps1_ravel = omp_get_wtime();
	this->one_D_to_three_D_same_dimensions(beta, params, 3*M.n_gauss, indice_y, indice_x);
	double temps2_ravel = omp_get_wtime();
	temps_ravel+=temps2_ravel-temps1_ravel;
	//unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

double temps1_tableaux = omp_get_wtime();
	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			std::vector<T> residual_1D(indice_v,0.);
			std::vector<T> params_flat(params.size(),0.);
			std::vector<T> cube_flat(cube[0][0].size(),0.);

			for (int p=0; p<3*M.n_gauss; p++){
				params_flat[p]=params[p][i][j];
			}
			for (int p=0; p<indice_v; p++){
				cube_flat[p]=cube[j][i][p];
			}

			this->myresidual(params_flat, cube_flat, residual_1D, M.n_gauss);

			for (int p=0;p<indice_v;p++){
				residual[p][i][j]=residual_1D[p];
			}

			if(std_map[i][j]>0.){
				f += this->myfunc_spec(residual_1D)*1/pow(std_map[i][j],2.); //std_map est arrondie... 
			}

		}
	}
double temps2_tableaux = omp_get_wtime();

double temps1_dF_dB = omp_get_wtime();
double temps2_dF_dB = omp_get_wtime();

double temps1_deriv = omp_get_wtime();
	int i;

	#pragma omp parallel private(i) shared(params,deriv,std_map,residual,indice_v,indice_y,indice_x)
	{
	#pragma omp for
	for(i=0; i<M.n_gauss; i++){
		for(int k=0; k<indice_v; k++){
			for(int j=0; j<indice_y; j++){
				for(int l=0; l<indice_x; l++){
					if(std_map[j][l]>0.){
						T spec = T(k+1);
						deriv[0+3*i][j][l] += exp(-pow( spec-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*residual[k][j][l]/pow(std_map[j][l],2);
						deriv[1+3*i][j][l] += params[3*i][j][l]*( spec - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( spec-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*residual[k][j][l]/pow(std_map[j][l],2);
						deriv[2+3*i][j][l] += params[3*i][j][l]*pow( spec - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( spec-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*residual[k][j][l]/pow(std_map[j][l],2);
					}
				}
			}
		}
	}
	}
	double temps2_deriv = omp_get_wtime();
	double temps1_conv = omp_get_wtime();

	for(int i=0; i<M.n_gauss; i++){
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){
				image_amp[p][q]=params[0+3*i][p][q];
				image_mu[p][q]=params[1+3*i][p][q];
				image_sig[p][q]=params[2+3*i][p][q];
			}
		}

		this->convolution_2D_mirror(M, image_amp, conv_amp, indice_y, indice_x,3);
		this->convolution_2D_mirror(M, image_mu, conv_mu, indice_y, indice_x,3);
		this->convolution_2D_mirror(M, image_sig, conv_sig, indice_y, indice_x,3);
	
		this->convolution_2D_mirror(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
		this->convolution_2D_mirror(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
		this->convolution_2D_mirror(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);

//		printf("M.n_gauss = %d , M.lambda_amp = %f, M.lambda_mu = %f , M.lambda_sig = %f\n", M.n_gauss, M.lambda_amp, M.lambda_mu, M.lambda_sig);

		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){
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
	double temps2_conv = omp_get_wtime();
	temps_tableaux += temps2_tableaux - temps1_tableaux;
	temps_ravel+=temps2_ravel-temps1_ravel;

	temps1_ravel = omp_get_wtime();
	this->three_D_to_one_D_same_dimensions(deriv, g, 3*M.n_gauss, indice_y, indice_x);
	temps2_ravel = omp_get_wtime();

	temps_ravel+=temps2_ravel-temps1_ravel;

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;

	temps_f_g_cube += temps2_dF_dB - temps1_dF_dB;
	}

template <typename T> void algo_rohsa::f_g_cube_fast_without_regul(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map){

	std::vector<std::vector<std::vector<T>>> deriv(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<std::vector<std::vector<T>>> g_3D(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));

	std::vector<std::vector<std::vector<T>>> residual(indice_x,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_v,0.)));

	std::vector<std::vector<std::vector<T>>> params(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<T> b_params(M.n_gauss,0.);

	std::vector<std::vector<T>> conv_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_sig(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> conv_conv_sig(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_amp(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_mu(indice_y,std::vector<T>(indice_x, 0.));
	std::vector<std::vector<T>> image_sig(indice_y,std::vector<T>(indice_x, 0.));

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
			std::vector<T> residual_1D(indice_v,0.);
			std::vector<T> params_flat(params.size(),0.);
			std::vector<T> cube_flat(cube[0][0].size(),0.);
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
				T v_1=0.,v_2=0.,v_3=0.;

			if(std_map[j][l]>0.){
	//	    #pragma omp parallel shared(M, params, residual, std_map, indice_v,i,j,l)
	//	    {
	//        #pragma omp for private(k)
			for(k=0; k<indice_v; k++){
				v_1 += exp(-pow( T(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
	//printf("--->curseur = %e\n", std_map[j][l] );//residual[l][j][k]  );
				v_2 +=  params[3*i][j][l]*( T(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( T(k+1)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
		
				v_3 += params[3*i][j][l]*pow( T(k+1) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( T(k+1)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
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
template <typename T> void algo_rohsa::f_g_cube_vector(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map){

//std::cout<<"Début f_g_cube"<<std::endl;


std::vector<std::vector<std::vector<T>>> dR_over_dB(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));

//décommenter en bas

std::vector<std::vector<std::vector<std::vector<T>>>> dF_over_dB(3*M.n_gauss,std::vector<std::vector<std::vector<T>>>( indice_v,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.))));
std::vector<std::vector<std::vector<T>>> deriv(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
std::vector<std::vector<std::vector<T>>> g_3D(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));

std::vector<std::vector<std::vector<T>>> residual(indice_x,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_v,0.)));

std::vector<std::vector<std::vector<T>>> params(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
std::vector<T> b_params(M.n_gauss,0.);

std::vector<std::vector<T>> conv_amp(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> conv_mu(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> conv_sig(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> conv_conv_amp(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> conv_conv_mu(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> conv_conv_sig(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> image_amp(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> image_mu(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> image_sig(indice_y,std::vector<T>(indice_x, 0.));

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
		std::vector<T> residual_1D(indice_v,0.);
		std::vector<T> params_flat(params.size(),0.);
		std::vector<T> cube_flat(cube[0][0].size(),0.);
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

#pragma omp parallel private(i,k) shared(dF_over_dB,params,M,indice_v,indice_y,indice_x) //num_threads(1) //
{
#pragma omp for
for(int i=0; i<M.n_gauss; i++){
        for(k=0; k<indice_v; k++){
                for(int j=0; j<indice_y; j++){
                        for(int l=0; l<indice_x; l++){

				dF_over_dB[0+3*i][k][j][l] += exp(-pow( T(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

				dF_over_dB[1+3*i][k][j][l] +=  params[3*i][j][l]*( T(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * 
									exp(-pow( T(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

				dF_over_dB[2+3*i][k][j][l] += params[3*i][j][l]*pow( T(k+1) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * 
									exp(-pow( T(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

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



//test optims OpenMP for f_g_cube
template <typename T> void algo_rohsa::f_g_cube_cuda_L(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened) 
{

std::vector<std::vector<std::vector<T>>> dR_over_dB(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
//décommenter en bas

std::vector<std::vector<std::vector<T>>> g_3D(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));

std::vector<T> b_params(M.n_gauss,0.);

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

	size_t size_deriv = product_deriv * sizeof(T);
	size_t size_res = product_residual * sizeof(T);
	size_t size_std = product_std_map_ * sizeof(T);
	size_t size_beta_modif = product_beta_modif * sizeof(T);
	size_t size_image_conv = product_image_conv * sizeof(T);

	T* deriv = (T*)malloc(size_deriv);
	T* residual = (T*)malloc(size_res);
	T* std_map_ = (T*)malloc(size_std);
	T* beta_modif = (T*)malloc(size_beta_modif);

	T* conv_amp = (T*)malloc(size_image_conv);
	T* conv_mu = (T*)malloc(size_image_conv);
	T* conv_sig = (T*)malloc(size_image_conv);
	T* conv_conv_amp = (T*)malloc(size_image_conv);
	T* conv_conv_mu = (T*)malloc(size_image_conv);
	T* conv_conv_sig = (T*)malloc(size_image_conv);
	T* image_amp = (T*)malloc(size_image_conv);
	T* image_mu = (T*)malloc(size_image_conv);
	T* image_sig = (T*)malloc(size_image_conv);


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

	double temps1_deriv = omp_get_wtime();

	gradient_L_2_beta(deriv, taille_deriv, product_deriv, beta_modif, taille_beta_modif, product_beta_modif, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, M.n_gauss);
/*
	for(int i = 0; i<10; i++){
		printf("deriv[%d]= %f \n",i, deriv[i]);
	}
*/
//printf("deriv[0] = %f", deriv[0]);
//exit(0);

    T Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};

//	printf("deriv[0] = %f \n",deriv[0]);

	double temps2_deriv = omp_get_wtime();
	double temps1_conv = omp_get_wtime();

	for(int k=0; k<M.n_gauss; k++){
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){
				image_amp[p+indice_x*q]= beta[q*indice_y*3*M.n_gauss+ p*3*M.n_gauss+ (0+3*k)];
				image_mu[p+indice_x*q]=beta[q*indice_y*3*M.n_gauss+ p*3*M.n_gauss+ (1+3*k)];
				image_sig[p+indice_x*q]=beta[q*indice_y*3*M.n_gauss+ p*3*M.n_gauss+ (2+3*k)];
			}
		}

if(false){//indice_x>=128 || indice_y>=128){
	conv2D_GPU(image_amp, Kernel, conv_amp, indice_x, indice_y, temps_transfert, temps_mirroirs);
	conv2D_GPU(image_mu, Kernel, conv_mu, indice_x, indice_y, temps_transfert, temps_mirroirs);
	conv2D_GPU(image_sig, Kernel, conv_sig, indice_x, indice_y, temps_transfert, temps_mirroirs);

	conv2D_GPU(conv_amp, Kernel, conv_conv_amp, indice_x, indice_y, temps_transfert, temps_mirroirs);
	conv2D_GPU(conv_mu, Kernel, conv_conv_mu, indice_x, indice_y, temps_transfert, temps_mirroirs);
	conv2D_GPU(conv_sig, Kernel, conv_conv_sig, indice_x, indice_y, temps_transfert, temps_mirroirs);

} else{
	convolution_2D_mirror_flat(M, image_amp, conv_amp, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
	convolution_2D_mirror_flat(M, image_mu, conv_mu, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
	convolution_2D_mirror_flat(M, image_sig, conv_sig, indice_y, indice_x,3, temps_transfert, temps_mirroirs);

	convolution_2D_mirror_flat(M, conv_amp, conv_conv_amp, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
	convolution_2D_mirror_flat(M, conv_mu, conv_conv_mu, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
	convolution_2D_mirror_flat(M, conv_sig, conv_conv_sig, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
}

	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			f+= 0.5*M.lambda_amp*pow(conv_amp[i+indice_x*j],2);
			f+= 0.5*M.lambda_mu*pow(conv_mu[i+indice_x*j],2);
			f+= 0.5*M.lambda_sig*pow(conv_sig[i+indice_x*j],2) + 0.5*M.lambda_var_sig*pow(image_sig[i+indice_x*j]-b_params[k],2);

			deriv[(0+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_amp*conv_conv_amp[i+indice_x*j];
			deriv[(1+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_mu*conv_conv_mu[i+indice_x*j];

			deriv[(2+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_sig*conv_conv_sig[i+indice_x*j]+M.lambda_var_sig*(image_sig[i+indice_x*j]-b_params[k]);
			g[n_beta-M.n_gauss+k] += M.lambda_var_sig*(b_params[k]-image_sig[i+indice_x*j]);
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





template <typename T> void algo_rohsa::f_g_cube_cuda_L_clean(parameters &M, T& f, T* g, int n, std::vector<std::vector<std::vector<T>>> &cube, T* beta, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened) 
{
	bool print = false;	
	int lim = 100;
/*
    if(indice_x>=256){
        print = true;
    }
*/
	if(print){
		printf("Début :\n");
		for(int i=0; i<lim; i++){
			printf("beta[%d] = %.16f\n",i, beta[i]);
		}
		printf("f = %.16f\n",f);
		std::cin.ignore();
	}
    
	std::vector<T> b_params(M.n_gauss,0.);
	int i,k,j,l,p;

	int taille_params_flat[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_deriv[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_residual[] = {indice_v, indice_y, indice_x};
	int taille_std_map_[] = {indice_y, indice_x};
	int taille_beta[] = {3*M.n_gauss, indice_y, indice_x};
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

	size_t size_deriv = product_deriv * sizeof(T);
	size_t size_res = product_residual * sizeof(T);
	size_t size_std = product_std_map_ * sizeof(T);
	size_t size_beta_modif = product_beta_modif * sizeof(T);
	size_t size_image_conv = product_image_conv * sizeof(T);

	T* deriv = (T*)malloc(size_deriv);
	T* residual = (T*)malloc(size_res);
	T* std_map_ = (T*)malloc(size_std);

	T* conv_amp = (T*)malloc(size_image_conv);
	T* conv_mu = (T*)malloc(size_image_conv);
	T* conv_sig = (T*)malloc(size_image_conv);
	T* conv_conv_amp = (T*)malloc(size_image_conv);
	T* conv_conv_mu = (T*)malloc(size_image_conv);
	T* conv_conv_sig = (T*)malloc(size_image_conv);
	T* image_amp = (T*)malloc(size_image_conv);
	T* image_mu = (T*)malloc(size_image_conv);
	T* image_sig = (T*)malloc(size_image_conv);


	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;

	double temps1_ravel = omp_get_wtime();
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
//		printf("b_params[%d]= %f\n", i, b_params[i]);
	}
	double temps2_ravel = omp_get_wtime();
	temps_ravel+=temps2_ravel-temps1_ravel;


//exit(0);
	double temps_modification_beta1 = omp_get_wtime();

	double temps_modification_beta2 = omp_get_wtime();

	double temps1_dF_dB = omp_get_wtime();
	double temps2_dF_dB = omp_get_wtime();

	double temps1_tableaux = omp_get_wtime();
	f =  compute_residual_and_f(beta, taille_beta_modif, product_beta, cube_flattened, taille_cube, product_cube, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, indice_x, indice_y, indice_v, M.n_gauss);
	double temps2_tableaux = omp_get_wtime();

	double temps1_deriv = omp_get_wtime();
	gradient_L_2_beta(deriv, taille_deriv, product_deriv, beta, taille_beta_modif, product_beta_modif, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, M.n_gauss);
	double temps2_deriv = omp_get_wtime();

	if(print){
		printf("Milieu :\n");
		for(int i=0; i<lim; i++){
			printf("deriv[%d] = %.16f\n",i, deriv[i]);
		}
		printf("-> f = %.16f\n", f);
		std::cin.ignore();
	}

    T Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};

	double temps1_conv = omp_get_wtime();

	for(int k=0; k<M.n_gauss; k++){
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){
				image_amp[indice_x*p+q]= beta[(0+3*k)*indice_x*indice_y + p*indice_x+q];
				image_mu[indice_x*p+q]=beta[(1+3*k)*indice_x*indice_y + p*indice_x+q];
				image_sig[indice_x*p+q]=beta[(2+3*k)*indice_x*indice_y + p*indice_x+q];
			}
		}
/*
		if(print){
			for(int p=0; p<4; p++){
				printf("image_sig[%d] = %.16f\n", p, image_sig[p]);
			}
		}
*/
		if(false){//indice_x>=128 || indice_y>=128){//true){//
			conv2D_GPU(image_amp, Kernel, conv_amp, indice_x, indice_y, temps_transfert, temps_mirroirs);
			conv2D_GPU(image_mu, Kernel, conv_mu, indice_x, indice_y, temps_transfert, temps_mirroirs);
			conv2D_GPU(image_sig, Kernel, conv_sig, indice_x, indice_y, temps_transfert, temps_mirroirs);

			conv2D_GPU(conv_amp, Kernel, conv_conv_amp, indice_x, indice_y, temps_transfert, temps_mirroirs);
			conv2D_GPU(conv_mu, Kernel, conv_conv_mu, indice_x, indice_y, temps_transfert, temps_mirroirs);
			conv2D_GPU(conv_sig, Kernel, conv_conv_sig, indice_x, indice_y, temps_transfert, temps_mirroirs);
		} else{
			convolution_2D_mirror_flat(M, image_amp, conv_amp, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
			convolution_2D_mirror_flat(M, image_mu, conv_mu, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
			convolution_2D_mirror_flat(M, image_sig, conv_sig, indice_y, indice_x,3, temps_transfert, temps_mirroirs);

			convolution_2D_mirror_flat(M, conv_amp, conv_conv_amp, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
			convolution_2D_mirror_flat(M, conv_mu, conv_conv_mu, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
			convolution_2D_mirror_flat(M, conv_sig, conv_conv_sig, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
		}
		T f_1 = 0.;
		T f_2 = 0.;
		T f_3 = 0.;
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				f_1+= 0.5*M.lambda_amp*pow(conv_amp[indice_x*i+j],2);
				f_2+= 0.5*M.lambda_mu*pow(conv_mu[indice_x*i+j],2);
				f_3+= 0.5*M.lambda_sig*pow(conv_sig[indice_x*i+j],2) + 0.5*M.lambda_var_sig*pow(image_sig[indice_x*i+j]-b_params[k],2);

				deriv[(0+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_amp*conv_conv_amp[indice_x*i+j];
				deriv[(1+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_mu*conv_conv_mu[indice_x*i+j];
				deriv[(2+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_sig*conv_conv_sig[indice_x*i+j]+M.lambda_var_sig*(image_sig[indice_x*i+j]-b_params[k]);
				g[n_beta-M.n_gauss+k] += M.lambda_var_sig*(b_params[k]-image_sig[indice_x*i+j]);
			}
		}

	if(print){
/*		for(int p=0; p<300; p++){
			printf("image_amp[%d] = %.16f\n", p, image_amp[p]);
		}
		for(int p=0; p<300; p++){
			printf("image_mu[%d] = %.16f\n", p, image_mu[p]);
		}
		for(int p=0; p<300; p++){
			printf("image_sig[%d] = %.16f\n", p, image_sig[p]);
		}
		for(int p=0; p<300; p++){
			printf("conv_mu[%d] = %.16f\n", p, conv_mu[p]);
		}
		for(int p=0; p<300; p++){
			printf("conv_sig[%d] = %.16f\n", p, conv_sig[p]);
		}
		*/
		for(int p=0; p<300; p++){
			printf("conv_amp[%d] = %.16f\n", p, conv_amp[p]);
		}
	    printf("Début print f_1 : %.16f\n", f_1);
		for(int p=0; p<15; p++){
			printf("conv_amp[%d] = %.16f\n", p, conv_amp[p]);
		}
	    printf("Début print f_2 : %.16f\n", f_2);
		for(int p=0; p<15; p++){
			printf("conv_mu[%d] = %.16f\n", p, conv_mu[p]);
		}
	    printf("Début print f_3 : %.16f\n", f_3);
		for(int p=0; p<15; p++){
			printf("conv_sig[%d] = %.16f\n", p, conv_sig[p]);
		}
		std::cin.ignore();
		}
		f+=f_1+f_2+f_3;
	}
	double temps2_conv = omp_get_wtime();

	temps_ravel+=temps2_ravel-temps1_ravel;
	temps1_ravel = omp_get_wtime();
	for(int i=0; i<n_beta-M.n_gauss; i++){
		g[i]=deriv[i];
	}
	temps2_ravel = omp_get_wtime();
	temps_ravel+=temps2_ravel-temps1_ravel;


	if (print)
	{
		for(int i=0; i<lim; i++){
			printf("g[%d] = %.16f\n",i, g[i]);
		}
		for(int i=lim-40; i<lim; i++){
			printf("g[%d] = %.16f\n",i, g[i]);
		}
		printf("fin -> f = %.16f\n", f);
        std::cin.ignore();
	}



	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;

	temps_modification_beta += temps_modification_beta2 - temps_modification_beta1;

	temps_f_g_cube += temps2_dF_dB - temps1_dF_dB;

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



template <typename T> void algo_rohsa::f_g_cube_cuda_L_clean_lib(parameters &M, T &f, T* g, int n, T* beta, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened) 
{

	std::vector<T> b_params(M.n_gauss,0.);

	int i,k,j,l,p;

	int taille_params_flat[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_deriv[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_residual[] = {indice_v, indice_y, indice_x};
	int taille_std_map_[] = {indice_y, indice_x};
	int taille_beta[] = {3*M.n_gauss, indice_y, indice_x};
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

	size_t size_deriv = product_deriv * sizeof(T);
	size_t size_res = product_residual * sizeof(T);
	size_t size_std = product_std_map_ * sizeof(T);
	size_t size_beta_modif = product_beta_modif * sizeof(T);
	size_t size_image_conv = product_image_conv * sizeof(T);

	T* deriv = (T*)malloc(size_deriv);
	T* residual = (T*)malloc(size_res);
	T* std_map_ = (T*)malloc(size_std);

	T* conv_amp = (T*)malloc(size_image_conv);
	T* conv_mu = (T*)malloc(size_image_conv);
	T* conv_sig = (T*)malloc(size_image_conv);
	T* conv_conv_amp = (T*)malloc(size_image_conv);
	T* conv_conv_mu = (T*)malloc(size_image_conv);
	T* conv_conv_sig = (T*)malloc(size_image_conv);
	T* image_amp = (T*)malloc(size_image_conv);
	T* image_mu = (T*)malloc(size_image_conv);
	T* image_sig = (T*)malloc(size_image_conv);

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;

	double temps1_ravel = omp_get_wtime();
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

	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
//		printf("b_params[%d]= %f\n", i, b_params[i]);
	}
	double temps2_ravel = omp_get_wtime();
	temps_ravel+=temps2_ravel-temps1_ravel;
	double temps_modification_beta1 = omp_get_wtime();

	double temps_modification_beta2 = omp_get_wtime();

	double temps1_dF_dB = omp_get_wtime();
	double temps2_dF_dB = omp_get_wtime();

	double temps1_tableaux = omp_get_wtime();
	f =  compute_residual_and_f(beta, taille_beta_modif, product_beta, cube_flattened, taille_cube, product_cube, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, indice_x, indice_y, indice_v, M.n_gauss);
	double temps2_tableaux = omp_get_wtime();

	double temps1_deriv = omp_get_wtime();
	gradient_L_2_beta(deriv, taille_deriv, product_deriv, beta, taille_beta_modif, product_beta_modif, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, M.n_gauss);
	double temps2_deriv = omp_get_wtime();

    T Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};

	double temps1_conv = omp_get_wtime();

	for(int k=0; k<M.n_gauss; k++){
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){
				image_amp[indice_x*p+q]= beta[(0+3*k)*indice_x*indice_y + p*indice_x+q];
				image_mu[indice_x*p+q]=beta[(1+3*k)*indice_x*indice_y + p*indice_x+q];
				image_sig[indice_x*p+q]=beta[(2+3*k)*indice_x*indice_y + p*indice_x+q];
			}
		}

		if(false){//indice_x>=128 || indice_y>=128){//true){//
			conv2D_GPU(image_amp, Kernel, conv_amp, indice_x, indice_y, temps_transfert, temps_mirroirs);
			conv2D_GPU(image_mu, Kernel, conv_mu, indice_x, indice_y, temps_transfert, temps_mirroirs);
			conv2D_GPU(image_sig, Kernel, conv_sig, indice_x, indice_y, temps_transfert, temps_mirroirs);

			conv2D_GPU(conv_amp, Kernel, conv_conv_amp, indice_x, indice_y, temps_transfert, temps_mirroirs);
			conv2D_GPU(conv_mu, Kernel, conv_conv_mu, indice_x, indice_y, temps_transfert, temps_mirroirs);
			conv2D_GPU(conv_sig, Kernel, conv_conv_sig, indice_x, indice_y, temps_transfert, temps_mirroirs);
		} else{
			this->convolution_2D_mirror_flat(M, image_amp, conv_amp, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
			this->convolution_2D_mirror_flat(M, image_mu, conv_mu, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
			this->convolution_2D_mirror_flat(M, image_sig, conv_sig, indice_y, indice_x,3, temps_transfert, temps_mirroirs);

			this->convolution_2D_mirror_flat(M, conv_amp, conv_conv_amp, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
			this->convolution_2D_mirror_flat(M, conv_mu, conv_conv_mu, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
			this->convolution_2D_mirror_flat(M, conv_sig, conv_conv_sig, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
		}

		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				f+= 0.5*M.lambda_amp*pow(conv_amp[indice_x*i+j],2);
				f+= 0.5*M.lambda_mu*pow(conv_mu[indice_x*i+j],2);
				f+= 0.5*M.lambda_sig*pow(conv_sig[indice_x*i+j],2) + 0.5*M.lambda_var_sig*pow(image_sig[indice_x*i+j]-b_params[k],2);

				deriv[(0+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_amp*conv_conv_amp[indice_x*i+j];
				deriv[(1+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_mu*conv_conv_mu[indice_x*i+j];
				deriv[(2+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_sig*conv_conv_sig[indice_x*i+j]+M.lambda_var_sig*(image_sig[indice_x*i+j]-b_params[k]);
				g[n_beta-M.n_gauss+k] += M.lambda_var_sig*(b_params[k]-image_sig[indice_x*i+j]);
			}
		}
	}
	double temps2_conv = omp_get_wtime();

	temps_ravel+=temps2_ravel-temps1_ravel;
	temps1_ravel = omp_get_wtime();
	for(int i=0; i<n_beta-M.n_gauss; i++){
		g[i]=deriv[i];
	}
	temps2_ravel = omp_get_wtime();
	temps_ravel+=temps2_ravel-temps1_ravel;

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;

	temps_modification_beta += temps_modification_beta2 - temps_modification_beta1;

	temps_f_g_cube += temps2_dF_dB - temps1_dF_dB;

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



template <typename T> void algo_rohsa::f_g_cube_cuda_L_clean_lib_device_fg(parameters &M, T &f, T* g_dev, int n, T* beta_dev, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened) 
{
//	printf("TEST 1\n");

    T* g = NULL;
    g = (T*)malloc(n*sizeof(T)); 
    T* beta = NULL;
    beta = (T*)malloc(n*sizeof(T)); 

	checkCudaErrors(cudaMemcpy(g, g_dev, n*sizeof(T), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(beta, beta_dev, n*sizeof(T), cudaMemcpyDeviceToHost));

	std::vector<T> b_params(M.n_gauss,0.);

	int i,k,j,l,p;

	int taille_params_flat[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_deriv[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_residual[] = {indice_v, indice_y, indice_x};
	int taille_std_map_[] = {indice_y, indice_x};
	int taille_beta[] = {3*M.n_gauss, indice_y, indice_x};
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

	size_t size_deriv = product_deriv * sizeof(T);
	size_t size_res = product_residual * sizeof(T);
	size_t size_std = product_std_map_ * sizeof(T);
	size_t size_beta_modif = product_beta_modif * sizeof(T);
	size_t size_image_conv = product_image_conv * sizeof(T);

	T* deriv = (T*)malloc(size_deriv);
	T* residual = (T*)malloc(size_res);
	T* std_map_ = (T*)malloc(size_std);

	T* conv_amp = (T*)malloc(size_image_conv);
	T* conv_mu = (T*)malloc(size_image_conv);
	T* conv_sig = (T*)malloc(size_image_conv);
	T* conv_conv_amp = (T*)malloc(size_image_conv);
	T* conv_conv_mu = (T*)malloc(size_image_conv);
	T* conv_conv_sig = (T*)malloc(size_image_conv);
	T* image_amp = (T*)malloc(size_image_conv);
	T* image_mu = (T*)malloc(size_image_conv);
	T* image_sig = (T*)malloc(size_image_conv);

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;

	double temps1_ravel = omp_get_wtime();

//	printf("TEST 2\n");

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

	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
//		printf("b_params[%d]= %f\n", i, b_params[i]);
	}
//	printf("TEST 3\n");

	double temps2_ravel = omp_get_wtime();
	temps_ravel+=temps2_ravel-temps1_ravel;
	double temps_modification_beta1 = omp_get_wtime();

	double temps_modification_beta2 = omp_get_wtime();

	double temps1_dF_dB = omp_get_wtime();
	double temps2_dF_dB = omp_get_wtime();

	double temps1_tableaux = omp_get_wtime();
	f =  compute_residual_and_f(beta, taille_beta_modif, product_beta, cube_flattened, taille_cube, product_cube, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, indice_x, indice_y, indice_v, M.n_gauss);
	double temps2_tableaux = omp_get_wtime();

	double temps1_deriv = omp_get_wtime();
	gradient_L_2_beta(deriv, taille_deriv, product_deriv, beta, taille_beta_modif, product_beta_modif, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, M.n_gauss);
	double temps2_deriv = omp_get_wtime();

    T Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};

	double temps1_conv = omp_get_wtime();

//	printf("TEST 4\n");

	for(int k=0; k<M.n_gauss; k++){
//		printf("TEST 5\n");
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){
				image_amp[indice_x*p+q]= beta[(0+3*k)*indice_x*indice_y + p*indice_x+q];
				image_mu[indice_x*p+q]=beta[(1+3*k)*indice_x*indice_y + p*indice_x+q];
				image_sig[indice_x*p+q]=beta[(2+3*k)*indice_x*indice_y + p*indice_x+q];
			}
		}
//		printf("TEST 6\n");

		if(false){//indice_x>=128 || indice_y>=128){//true){//
			conv2D_GPU(image_amp, Kernel, conv_amp, indice_x, indice_y, temps_transfert, temps_mirroirs);
			conv2D_GPU(image_mu, Kernel, conv_mu, indice_x, indice_y, temps_transfert, temps_mirroirs);
			conv2D_GPU(image_sig, Kernel, conv_sig, indice_x, indice_y, temps_transfert, temps_mirroirs);

			conv2D_GPU(conv_amp, Kernel, conv_conv_amp, indice_x, indice_y, temps_transfert, temps_mirroirs);
			conv2D_GPU(conv_mu, Kernel, conv_conv_mu, indice_x, indice_y, temps_transfert, temps_mirroirs);
			conv2D_GPU(conv_sig, Kernel, conv_conv_sig, indice_x, indice_y, temps_transfert, temps_mirroirs);
		} else{
			this->convolution_2D_mirror_flat(M, image_amp, conv_amp, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
			this->convolution_2D_mirror_flat(M, image_mu, conv_mu, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
			this->convolution_2D_mirror_flat(M, image_sig, conv_sig, indice_y, indice_x,3, temps_transfert, temps_mirroirs);

			this->convolution_2D_mirror_flat(M, conv_amp, conv_conv_amp, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
			this->convolution_2D_mirror_flat(M, conv_mu, conv_conv_mu, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
			this->convolution_2D_mirror_flat(M, conv_sig, conv_conv_sig, indice_y, indice_x,3, temps_transfert, temps_mirroirs);
		}

//		printf("TEST 7\n");
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				f+= 0.5*M.lambda_amp*pow(conv_amp[indice_x*i+j],2);
				f+= 0.5*M.lambda_mu*pow(conv_mu[indice_x*i+j],2);
				f+= 0.5*M.lambda_sig*pow(conv_sig[indice_x*i+j],2) + 0.5*M.lambda_var_sig*pow(image_sig[indice_x*i+j]-b_params[k],2);

				deriv[(0+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_amp*conv_conv_amp[indice_x*i+j];
				deriv[(1+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_mu*conv_conv_mu[indice_x*i+j];
				deriv[(2+3*k)*indice_y*indice_x+i*indice_x+j] += M.lambda_sig*conv_conv_sig[indice_x*i+j]+M.lambda_var_sig*(image_sig[indice_x*i+j]-b_params[k]);
				g[n_beta-M.n_gauss+k] += M.lambda_var_sig*(b_params[k]-image_sig[indice_x*i+j]);
			}
		}
	}
	double temps2_conv = omp_get_wtime();
//	printf("TEST 8\n");

	temps_ravel+=temps2_ravel-temps1_ravel;
	temps1_ravel = omp_get_wtime();
	for(int i=0; i<n_beta-M.n_gauss; i++){
		g[i]=deriv[i];
	}
	temps2_ravel = omp_get_wtime();
	temps_ravel+=temps2_ravel-temps1_ravel;

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;

	temps_modification_beta += temps_modification_beta2 - temps_modification_beta1;

	temps_f_g_cube += temps2_dF_dB - temps1_dF_dB;

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

	checkCudaErrors(cudaMemcpy(g_dev, g, n*sizeof(T), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(beta_dev, beta, n*sizeof(T), cudaMemcpyHostToDevice));
	free(g);
	free(beta);
}

template <typename T> void algo_rohsa::f_g_cube_omp_without_regul(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map) 
{

//std::cout<<"Début f_g_cube_test"<<std::endl;

std::vector<std::vector<std::vector<T>>> dR_over_dB(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
//°
//décommenter en bas
std::vector<std::vector<std::vector<std::vector<T>>>> dF_over_dB_bis(indice_v,std::vector<std::vector<std::vector<T>>>( indice_y,std::vector<std::vector<T>>(indice_x, std::vector<T>(3*M.n_gauss,0.))));
std::vector<std::vector<std::vector<T>>> deriv(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
//std::vector<std::vector<std::vector<T>>> deriv_bis(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
std::vector<std::vector<std::vector<T>>> g_3D(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
//dF_over_dB[v][y][x][M.n__gauss]
//deriv[gauss][y][x]
std::vector<std::vector<std::vector<T>>> residual(indice_x,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_v,0.)));

std::vector<std::vector<std::vector<T>>> params(3*M.n_gauss,std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
std::vector<std::vector<std::vector<T>>> params_T(indice_x,std::vector<std::vector<T>>(indice_y, std::vector<T>(3*M.n_gauss,0.)));
std::vector<T> b_params(M.n_gauss,0.);

std::vector<std::vector<T>> conv_amp(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> conv_mu(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> conv_sig(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> conv_conv_amp(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> conv_conv_mu(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> conv_conv_sig(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> image_amp(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> image_mu(indice_y,std::vector<T>(indice_x, 0.));
std::vector<std::vector<T>> image_sig(indice_y,std::vector<T>(indice_x, 0.));

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
		std::vector<T> residual_1D(indice_v,0.);
		std::vector<T> params_flat(params.size(),0.);
		std::vector<T> cube_flat(cube[0][0].size(),0.);
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
				T par0 = params_T[l][j][3*i+0];
				T par1 = T(k+1) - params_T[l][j][3*i+1];
				T par2 = params_T[l][j][3*i+2];

//				T par1_k = double(k+1) - par1;
				T par2_pow = 1/(2*pow(par2,2.));
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


template <typename T> void algo_rohsa::minimize_clean(parameters &M, long n, long m, T* beta, T* lb, T* ub, std::vector<std::vector<std::vector<T>>> &cube, std::vector<std::vector<T>> &std_map, int dim_x, int dim_y, int dim_v, T* cube_flattened) {
    int i__1;
	int  i__c = 0;
    double d__1, d__2;

    double t1, t2, f;

	T* g = (T*)malloc(n*sizeof(T));

    int i__;
    int taille_wa = 2*M.m*n+5*n+11*M.m*M.m+8*M.m;
    int taille_iwa = 3*n;

    long* nbd = NULL;
    nbd = (long*)malloc(n*sizeof(long)); 
    long* iwa = NULL;
    iwa = (long*)malloc(taille_iwa*sizeof(long)); 

	float temps_transfert_boucle = 0.;

    double* wa = NULL;
    wa = (double*)malloc(taille_wa*sizeof(double)); 

    long taskValue;
    long *task=&taskValue;

    double factr;
    long csaveValue;

    long *csave=&csaveValue;

    double dsave[29];
    long isave[44];
    logical lsave[4];
    double pgtol;

	double temps2_tableau_update = omp_get_wtime();

    for(int i(0); i<n; i++) {
	g[i]=0.;
    } 

	temps_tableau_update += omp_get_wtime() - temps2_tableau_update;

    f=0.;

    factr = 1e+7;
    pgtol = 1e-5;

    for (i__ = 0; i__ < n; i__ ++) {
        nbd[i__] = 2;
    }

    *task = (long)START;

	L111:

	double temps1_f_g_cube = omp_get_wtime();

    T* std_map_ = NULL;
    std_map_ = (T*)malloc(dim_x*dim_y*sizeof(T)); 


	for(int i=0; i<dim_y; i++){
		for(int j=0; j<dim_x; j++){
			std_map_[i*dim_x+j]=std_map[i][j];
		}
	}
	T* std_map_dev = NULL;
	checkCudaErrors(cudaMalloc(&std_map_dev, dim_x*dim_y*sizeof(T)));
	checkCudaErrors(cudaMemcpy(std_map_dev, std_map_, dim_x*dim_y*sizeof(T), cudaMemcpyHostToDevice));
	T* cube_flattened_dev = NULL;
	checkCudaErrors(cudaMalloc(&cube_flattened_dev, dim_x*dim_y*dim_v*sizeof(T)));
	checkCudaErrors(cudaMemcpy(cube_flattened_dev, cube_flattened, dim_x*dim_y*dim_v*sizeof(T), cudaMemcpyHostToDevice));

/*
	for(int i=0; i<200; i++){
		printf("--> minimize :: beta[%d] = %f\n",i, beta[i]);
	}
*/
	int compteur_iter_boucle_optim = 0;

	printf("dim_x = %d , dim_y = %d , dim_v = %d \n", dim_x, dim_y, dim_v);
	printf("n = %d , n_gauss = %d\n", int(n), M.n_gauss);
//	printf("beta[n_beta-1] = %f , beta[n_beta] = %f\n", beta[n-1], beta[n-1]);

	if (true){//dim_x >128){
		printf("beta[n_beta-1] = %f , beta[n_beta] = %f\n", beta[n-1], beta[n-1]);
		printf("cube_flattened[dim_x*dim_y*dim_v-1] = %f , cube_flattened[dim_x*dim_y*dim_v] = %f\n", cube_flattened[dim_x*dim_y*dim_v-1], cube_flattened[dim_x*dim_y*dim_v]);
	}

    T* temps = NULL;
    temps = (T*)malloc(4*sizeof(T)); 
    temps[0] = 0.;
    temps[1] = 0.;
    temps[2] = 0.;
    temps[3] = 0.;
    temps[4] = 0.;

	while(IS_FG(*task) or *task==NEW_X or *task==START){ 


		double temps_temp = omp_get_wtime();
		setulb(&n, &m, beta, lb, ub, nbd, &f, g, &factr, &pgtol, wa, iwa, task, 
				&M.iprint, csave, lsave, isave, dsave);
		temps_setulb += omp_get_wtime() - temps_temp;

		// CPU
//		f_g_cube_fast_clean<T>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map); //OK
//		f_g_cube_fast_clean_optim_CPU<T>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map); //OK
//		this->f_g_cube_fast_clean_optim_CPU_lib<T>(M, f, g, n, cube, beta, dim_v, dim_y, dim_x, std_map, NULL);

		// GPU
/*
		if(dim_x >128){
			for(int i = 0; i<300; i++){
				printf("beta[%d] = %f\n", i, beta[i]);
			}
		if(beta[0] ==  0.){
			for(int i = 0; i<300; i++){
				printf("cube_flattened[%d] = %f\n", i, cube_flattened[i]);
			}
		}
		}
*/

	if(false){//dim_x<64){
//		f_g_cube_fast_clean_optim_CPU<T>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map);
//		f_g_cube_cuda_L_clean<T>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened); // expérimentation gradient
	}else{

		if(M.select_version == 0){ //-cpu
			f_g_cube_fast_clean_optim_CPU<T>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map);
		}else if(M.select_version == 1){ //-gpu
			T* x_dev = nullptr;
			T* g_dev = nullptr;
			cudaMalloc(&x_dev, n * sizeof(x_dev[0]));
			cudaMalloc(&g_dev, n * sizeof(g_dev[0]));
			checkCudaErrors(cudaMemset(g_dev, 0., n*sizeof(g_dev[0])));
			checkCudaErrors(cudaMemcpy(x_dev, beta, n*sizeof(T), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaDeviceSynchronize());
			f_g_cube_parallel_lib<T>(M, f, g_dev, int(n), x_dev, int(dim_v), int(dim_y), int(dim_x), std_map_dev, cube_flattened_dev, temps);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaMemcpy(g, g_dev, n*sizeof(T), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaFree(x_dev));
			checkCudaErrors(cudaFree(g_dev));
			checkCudaErrors(cudaDeviceSynchronize());
		}else if(M.select_version == 2){ //-autre
			f_g_cube_cuda_L_clean<T>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened); // expérimentation gradient
		}
	}

		//dim g, beta : 3*M.n_gauss, indice_y, indice_x
//        this->f_g_cube_cuda_L_clean_lib<T>(M, f, g, n, beta, dim_v, dim_y, dim_x, std_map, cube_flattened);
 //		temps_transfert_boucle += f_g_cube_parallel_clean(M, f, g, n, cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened, this->temps_conv, this->temps_deriv, this->temps_tableaux, this->temps_f_g_cube, temps_transfert_boucle);
//		f_g_cube_parallel<T>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened, this->temps_conv, this->temps_deriv, this->temps_tableaux, this->temps_f_g_cube);
		//failed correction attempt
		//		f_g_cube_parallel<T>(M, f, g, n, cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened, this->temps_conv, this->temps_deriv, this->temps_tableaux, this->temps_f_g_cube, temps_transfert_boucle);

/*
	for(int i=0; i<50; i++){
		printf("g[%d] = %.16f\n",i, g[i]);
	}
        compteur_iter_boucle_optim++;
		printf("compteur_iter_boucle_optim = %d\n", compteur_iter_boucle_optim);
		printf(" --> fin-chemin : f = %.16f\n", f);
	    std::cin.ignore();
*/

/*
		if(dim_x!=dim_y){
			f_g_cube_fast_clean_optim_CPU<T>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map);
		}else{
			f_g_cube_cuda_L_clean<T>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened); 
		}
*/
/*
		int eight = 1;
		int nb_stacks = 1;
		printf("f = %.15f\n", f);
		for(int i = 0; i<n; i++){
			printf("--> %d, stack %d  g[%d] = %.15f\n",eight,nb_stacks, i, g[i]);
			eight ++;
			if (eight == 5){
				eight = 1;
				nb_stacks ++;
			}
		}
		std::cin.ignore();
*/
//		exit(0);

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
	for(int i=0; i<100; i++){
		printf("beta[%d] = %f\n",i, beta[i]);
	}
	printf("CLASSIQUE\n");
	exit(0);
	std::cin.ignore();
*/
///	printf("--> nombre d'itérations dans la boucle d'optimisation (limite à 800) = %d \n", compteur_iter_boucle_optim);


	double temps4_tableau_update = omp_get_wtime();
	
	temps_tableau_update += omp_get_wtime() - temps4_tableau_update;

	double temps2_f_g_cube = omp_get_wtime();

	this->temps_conv += temps[3]/1000;
	this->temps_deriv += temps[2]/1000;
	this->temps_tableaux += temps[1]/1000;
	this->temps_copy += temps[0]/1000;
	this->temps_f_g_cube +=  temps[4]/1000;

	printf("this->temps_copy = %f\n", this->temps_copy);
	printf("this->temps_tableaux = %f\n", this->temps_tableaux);
	printf("this->temps_deriv = %f\n", this->temps_deriv);
	printf("this->temps_conv = %f\n", this->temps_conv);
	printf("this->temps_f_g_cube = %f\n", this->temps_f_g_cube);

	std::cout<< "Temps de calcul gradient : " << temps2_f_g_cube - temps1_f_g_cube<<std::endl;

	free(wa);
	free(nbd);
	free(iwa);
	free(g);
	free(std_map_);
	free(temps);


	checkCudaErrors(cudaFree(std_map_dev));
	checkCudaErrors(cudaFree(cube_flattened_dev));


printf("----------------->temps_transfert_boucle = %f \n", temps_transfert_boucle);
this->temps_transfert_d += temps_transfert_boucle;
}



template <typename T> void algo_rohsa::minimize_clean_gpu(parameters &M, long n, long m, T* beta, T* lb, T* ub, std::vector<std::vector<std::vector<T>>> &cube, std::vector<std::vector<T>> &std_map, int dim_x, int dim_y, int dim_v, T* cube_flattened) {

    T* cube_flattened_dev = NULL;
    checkCudaErrors(cudaMalloc(&cube_flattened_dev, dim_x*dim_y*dim_v*sizeof(T)));
    checkCudaErrors(cudaMemcpy(cube_flattened_dev, cube_flattened, dim_x*dim_y*dim_v*sizeof(T), cudaMemcpyHostToDevice));

    T* std_map_ = NULL;
    std_map_ = (T*)malloc(dim_x*dim_y*sizeof(T)); 
	for(int i=0; i<dim_y; i++){
		for(int j=0; j<dim_x; j++){
			std_map_[i*dim_x+j]=std_map[i][j];
		}
	}
    T* std_map_dev = NULL;
    checkCudaErrors(cudaMalloc(&std_map_dev, dim_x*dim_y*sizeof(T)));
    checkCudaErrors(cudaMemcpy(std_map_dev, std_map_, dim_x*dim_y*sizeof(T), cudaMemcpyHostToDevice));
    double* temps = NULL;
    temps = (double*)malloc(4*sizeof(double)); 
    temps[0] = 0.;
    temps[1] = 0.;
    temps[2] = 0.;
    temps[3] = 0.;
    temps[4] = 0.;

    LBFGSB_CUDA_OPTION<T> lbfgsb_options;

    lbfgsbcuda::lbfgsbdefaultoption<T>(lbfgsb_options);
    lbfgsb_options.mode = LCM_CUDA;
    lbfgsb_options.eps_f = static_cast<T>(1e-8);
    lbfgsb_options.eps_g = static_cast<T>(1e-8);
    lbfgsb_options.eps_x = static_cast<T>(1e-8);
    lbfgsb_options.max_iteration = M.maxiter;

	// initialize LBFGSB state	
	LBFGSB_CUDA_STATE<T> state;
	memset(&state, 0, sizeof(state));
	T* assist_buffer_cuda = nullptr;
	cublasStatus_t stat = cublasCreate(&(state.m_cublas_handle));
	if (CUBLAS_STATUS_SUCCESS != stat) {
		std::cout << "CUBLAS init failed (" << stat << ")" << std::endl;
		exit(0);
	}

	T minimal_f = std::numeric_limits<T>::max();

  	// setup callback function that evaluate function value and its gradient
  	state.m_funcgrad_callback = [&assist_buffer_cuda, &minimal_f, this, &M, n, &cube_flattened_dev, &cube_flattened,
&std_map_dev, &std_map, dim_x, dim_y, dim_v, &temps, &beta](
                                  T* x_dev, T& f, T* g_dev,
                                  const cudaStream_t& stream,
                                  const LBFGSB_CUDA_SUMMARY<T>& summary) -> int {

		int temp = 0;
		for(int ind = 0; ind < n; ind++){
			if(isnan(beta[ind]) && temp == 0){
				checkCudaErrors(cudaMemcpy(x_dev, beta, n*sizeof(T), cudaMemcpyHostToDevice));
				temp=1;
			}
		}

	    checkCudaErrors(cudaDeviceSynchronize());
		f_g_cube_parallel_lib<T>(M, f, g_dev, n, x_dev, dim_v, dim_y, dim_x, std_map_dev, cube_flattened_dev, temps);
	    checkCudaErrors(cudaDeviceSynchronize());

		checkCudaErrors(cudaMemcpy(beta, x_dev, n*sizeof(T), cudaMemcpyDeviceToHost));


	
/*
        printf("LIB --> fin-chemin : f = %.16f\n", f);
	    std::cin.ignore();
	    checkCudaErrors(cudaDeviceSynchronize());
		f_g_cube_cuda_L_clean_lib_device_fg<T>(M, f, g_dev, n, x_dev, dim_v, dim_y, dim_x, std_map, cube_flattened); // expérimentation gradient
	    checkCudaErrors(cudaDeviceSynchronize());
*/
		//	f_g_cube_fast_clean_optim_CPU_lib<T>(M, f, g, n, cube, x, dim_v, dim_y, dim_x, std_map, &assist_buffer_cpu);
		//    dsscfg_cuda<T>(g_nx, g_ny, x, f, g, &assist_buffer_cuda, 'FG', g_lambda);
		if (summary.num_iteration % 100 == 0) {
		std::cout << "CUDA iteration " << summary.num_iteration << " F: " << f
					<< std::endl;
		}
		minimal_f = fmin(minimal_f, f);
//		printf("Before return\n", n);
		return 0;
	};

	// initialize CUDA buffers
	int N_elements = n;//g_nx * g_ny;

	T* x_dev = nullptr;
	T* g_dev = nullptr;
	T* xl_dev = nullptr;
	T* xu_dev = nullptr;
	int* nbd_dev = nullptr;

	printf("TEST GPU _________________________________\n");

	cudaMalloc(&x_dev, n * sizeof(x_dev[0]));
	cudaMalloc(&g_dev, n * sizeof(g_dev[0]));

	cudaMalloc(&xl_dev, n * sizeof(xl_dev[0]));
	cudaMalloc(&xu_dev, n * sizeof(xu_dev[0]));

	checkCudaErrors(cudaMemset(xl_dev, 0, n * sizeof(xl_dev[0])));
	cudaMemset(xu_dev, 0, n * sizeof(xu_dev[0]));

	checkCudaErrors(cudaMemcpy(x_dev, beta, n*sizeof(T), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(xl_dev, lb, n*sizeof(T), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(xu_dev, ub, n*sizeof(T), cudaMemcpyHostToDevice));

	// initialize starting point
	T f_init = 0.;
//	double f_init = std::numeric_limits<double>::max();
//dsscfg_cuda<T>(g_nx, g_ny, x, f_init, g, &assist_buffer_cuda, 'XS', g_lambda);
	// initialize number of bounds
	int* nbd = new int[n];
	memset(nbd, 0, n * sizeof(nbd[0]));
	for(int i = 0; i<n ; i++){
		nbd[i] = 2;		
	}
	cudaMalloc(&nbd_dev, n * sizeof(nbd_dev[0]));
	cudaMemset(nbd_dev, 0, n * sizeof(nbd_dev[0]));
	checkCudaErrors(cudaMemcpy(nbd_dev, nbd, n*sizeof(int), cudaMemcpyHostToDevice));

	LBFGSB_CUDA_SUMMARY<T> summary;
	memset(&summary, 0, sizeof(summary));

	double t1 = omp_get_wtime();
	printf("Before lbfgsbminimize\n");
	// call optimization
	auto start_time = std::chrono::steady_clock::now();
	lbfgsbcuda::lbfgsbminimize<T>(n, state, lbfgsb_options, x_dev, nbd_dev,
									xl_dev, xu_dev, summary);
	auto end_time = std::chrono::steady_clock::now();
	printf("After lbfgsbminimize\n");
	std::cout << "Timing: "
				<< (std::chrono::duration<T, std::milli>(end_time - start_time)
						.count() /
					static_cast<T>(summary.num_iteration))
				<< " ms / iteration" << std::endl;

	double t2 = omp_get_wtime();
	temps_global+=t2-t1;

	checkCudaErrors(cudaMemcpy(beta, x_dev, n*sizeof(T), cudaMemcpyDeviceToHost));

	this->temps_conv += temps[3]/1000;
	this->temps_deriv += temps[2]/1000;
	this->temps_tableaux += temps[1]/1000;
	this->temps_copy += temps[0]/1000;
	this->temps_f_g_cube +=  temps[4]/1000;

	printf("this->temps_copy = %f\n", this->temps_copy);
	printf("this->temps_tableaux = %f\n", this->temps_tableaux);
	printf("this->temps_deriv = %f\n", this->temps_deriv);
	printf("this->temps_conv = %f\n", this->temps_conv);
	printf("this->temps_f_g_cube = %f\n", this->temps_f_g_cube);

	std::cout<< "Temps de calcul gradient : " << t2 - t1<<std::endl;

	free(temps);
	// release allocated memory
	checkCudaErrors(cudaFree(x_dev));
	checkCudaErrors(cudaFree(g_dev));
	checkCudaErrors(cudaFree(xl_dev));
	checkCudaErrors(cudaFree(xu_dev));
	checkCudaErrors(cudaFree(nbd_dev));
	checkCudaErrors(cudaFree(std_map_dev));
	checkCudaErrors(cudaFree(cube_flattened_dev));
	delete[] nbd;
	checkCudaErrors(cudaFree(assist_buffer_cuda));
	free(std_map_);
	// release cublas
	cublasDestroy(state.m_cublas_handle);
//	return minimal_f;

	printf("TEST GPU _________________________________\n");

/*
	exit(0);
*/
}



template <typename T> void algo_rohsa::minimize_clean_cpu(parameters &M, long n, long m, T* beta, T* lb, T* ub, std::vector<std::vector<std::vector<T>>> &cube, std::vector<std::vector<T>> &std_map, int dim_x, int dim_y, int dim_v, T* cube_flattened) {

    double* temps = NULL;
    temps = (double*)malloc(4*sizeof(double)); 
    temps[0] = 0.;
    temps[1] = 0.;
    temps[2] = 0.;
    temps[3] = 0.;
    temps[4] = 0.;

    T* std_map_ = NULL;
    std_map_ = (T*)malloc(dim_x*dim_y*sizeof(T)); 


	for(int i=0; i<dim_y; i++){
		for(int j=0; j<dim_x; j++){
			std_map_[i*dim_x+j]=std_map[i][j];
		}
	}
	T* std_map_dev = NULL;
	checkCudaErrors(cudaMalloc(&std_map_dev, dim_x*dim_y*sizeof(T)));
	checkCudaErrors(cudaMemcpy(std_map_dev, std_map_, dim_x*dim_y*sizeof(T), cudaMemcpyHostToDevice));
	T* cube_flattened_dev = NULL;
	checkCudaErrors(cudaMalloc(&cube_flattened_dev, dim_x*dim_y*dim_v*sizeof(T)));
	checkCudaErrors(cudaMemcpy(cube_flattened_dev, cube_flattened, dim_x*dim_y*dim_v*sizeof(T), cudaMemcpyHostToDevice));

// we first initialize LBFGSB_CUDA_OPTION and LBFGSB_CUDA_STATE 
	LBFGSB_CUDA_OPTION<T> lbfgsb_options;

	lbfgsbcuda::lbfgsbdefaultoption<T>(lbfgsb_options);
	lbfgsb_options.mode = LCM_NO_ACCELERATION;
	lbfgsb_options.eps_f = static_cast<T>(1e-15);
	lbfgsb_options.eps_g = static_cast<T>(1e-15);
	lbfgsb_options.eps_x = static_cast<T>(1e-15);
	lbfgsb_options.max_iteration = M.maxiter;
    lbfgsb_options.step_scaling = 1.0;
	lbfgsb_options.hessian_approximate_dimension = 8;
  	lbfgsb_options.machine_epsilon = 1e-15;
  	lbfgsb_options.machine_maximum = std::numeric_limits<T>::max();

	// initialize LBFGSB state
	LBFGSB_CUDA_STATE<T> state;
	memset(&state, 0, sizeof(state));
	T* assist_buffer_cpu = nullptr;

  	T minimal_f = std::numeric_limits<T>::max();
  	state.m_funcgrad_callback = [&assist_buffer_cpu, &minimal_f, this, &M, &n, &cube, &cube_flattened, &cube_flattened_dev,
&std_map, &std_map_dev, dim_x, dim_y, dim_v, &temps](
                                  T* x, T& f, T* g,
                                  const cudaStream_t& stream,
                                  const LBFGSB_CUDA_SUMMARY<T>& summary) -> int {
		T* x_dev = nullptr;
		T* g_dev = nullptr;
		cudaMalloc(&x_dev, n * sizeof(x_dev[0]));
		cudaMalloc(&g_dev, n * sizeof(g_dev[0]));
		checkCudaErrors(cudaMemset(g_dev, 0., n*sizeof(g_dev[0])));
		checkCudaErrors(cudaMemcpy(x_dev, x, n*sizeof(T), cudaMemcpyHostToDevice));
    	checkCudaErrors(cudaDeviceSynchronize());
		f_g_cube_parallel_lib<T>(M, f, g_dev, int(n), x_dev, int(dim_v), int(dim_y), int(dim_x), std_map_dev, cube_flattened_dev, temps);
	    checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(g, g_dev, n*sizeof(T), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaFree(x_dev));
		checkCudaErrors(cudaFree(g_dev));
	    checkCudaErrors(cudaDeviceSynchronize());	//	f_g_cube_cuda_L_clean(M, f, g, n, cube, x, dim_v, dim_y, dim_x, std_map, cube_flattened);
	//  f_g_cube_cuda_L_clean<T>(M, f, g, &n, &cube, x, &dim_v, &dim_y, &dim_x, &std_map, &cube_flattened);
	//	this->f_g_cube_cuda_L_clean_lib<T>(M, f, g, n, x, dim_v, dim_y, dim_x, std_map, cube_flattened); // expérimentation gradient
    //    callback_test(M, f, g, n, cube, x, dim, std_map, &assist_buffer_cpu);
	//	this->f_g_cube_fast_clean_optim_CPU_lib<T>(M, f, g, n, cube, x, dim_v, dim_y, dim_x, std_map, &assist_buffer_cpu);
	//	f_g_cube_fast_clean_optim_CPU<T>(M, f, g, n, cube, x, dim_v, dim_y, dim_x, std_map);
		////dsscfg_cpu<T>(g_nx, g_ny, x, f, g, &assist_buffer_cpu, 'FG', g_lambda);
		if (summary.num_iteration % 100 == 0) {
		std::cout << "CPU iteration " << summary.num_iteration << " F: " << f
					<< std::endl;    
		}

		minimal_f = fmin(minimal_f, f);
//        minimal_f = f;

/*	
	for(int i=0; i<n; i++){
		printf("x[%d] = %.16f\n",i, x[i]);
	}
	for(int i=0; i<n; i++){
		printf("g[%d] = %.16f\n",i, g[i]);
	}
        printf("LIB --> fin-chemin : f = %.16f\n", f);
	    std::cin.ignore();
*/
		return 0;
	};
	// initialize CPU buffers
	int N_elements = n;
	T* x = new T[n];
	T* g = new T[n];

	T* xl = new T[n];
	T* xu = new T[n];

	// we have boundaries
	memset(xl, 0, n * sizeof(xl[0]));
	memset(xu, 0, n * sizeof(xu[0]));
	int* nbd = new int[n];
	memset(nbd, 0, n * sizeof(nbd[0]));

	for(int i = 0; i<n ; i++){
		x[i]=beta[i];
		xl[i] = lb[i];
		xu[i] = ub[i];
		nbd[i] = 2;		
	}


	// initialize starting point
	T f_init = 0.;
	//	f_g_cube_fast_clean_optim_CPU<T>(M, f_init, nullptr, n, cube, beta, dim_v, dim_y, dim_x, std_map);
////	dsscfg_cpu<T>(g_nx, g_ny, x, f_init, nullptr, &assist_buffer_cpu, 'XS', g_lambda);
	// initialize number of bounds (0 for this example)

	LBFGSB_CUDA_SUMMARY<T> summary;
	memset(&summary, 0, sizeof(summary));


	// call optimization
	auto start_time = std::chrono::steady_clock::now();
//	lbfgsbcuda::lbfgsbminimize<T>(n, state, lbfgsb_options, x, nbd,
//									xl, xu, summary);

	double t1 = omp_get_wtime();

	lbfgsbcuda::lbfgsbminimize<T>(n, state, lbfgsb_options, x, nbd,
									xl, xu, summary);

	double t2 = omp_get_wtime();

	auto end_time = std::chrono::steady_clock::now();
	std::cout << "Timing: "
				<< (std::chrono::duration<T, std::milli>(end_time - start_time)
						.count() /
					static_cast<T>(summary.num_iteration))
				<< " ms / iteration" << std::endl;


	for(int i = 0; i<n ; i++){
		beta[i]=x[i];
	}
	temps_global+=t2-t1;

	printf("temps_global cumulé = %f\n",temps_global);

/*
	for(int i=0; i<100; i++){
		printf("beta[%d] = %f\n",i, beta[i]);
	}
	printf("LIB\n");
	std::cin.ignore();
*/
	// release allocated memory
	delete[] x;
	delete[] g;
	delete[] xl;
	delete[] xu;
	delete[] nbd;
	delete[] assist_buffer_cpu;
	free(std_map_);
	checkCudaErrors(cudaFree(cube_flattened_dev));
	checkCudaErrors(cudaFree(std_map_dev));

	this->temps_conv += temps[3]/1000;
	this->temps_deriv += temps[2]/1000;
	this->temps_tableaux += temps[1]/1000;
	this->temps_copy += temps[0]/1000;
	this->temps_f_g_cube +=  temps[4]/1000;

	printf("this->temps_copy = %f\n", this->temps_copy);
	printf("this->temps_tableaux = %f\n", this->temps_tableaux);
	printf("this->temps_deriv = %f\n", this->temps_deriv);
	printf("this->temps_conv = %f\n", this->temps_conv);
	printf("this->temps_f_g_cube = %f\n", this->temps_f_g_cube);

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
        for(i=0;i<power; i++){ //dim_x
                for(j=0;j<power; j++){ //dim_y
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
//	std::cout<<"dim_v = "<<dim_v<<std::endl;
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
    std::vector<double> _residual_(line_v.size(),0.);
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

		if ( IS_FG(*task) ) {
			myresidual(x, line, _residual_, n_gauss_i);
			f = myfunc_spec(_residual_);
			mygrad_spec(g, _residual_, x, n_gauss_i);
//			printf("f = %f\n", f);
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

	for(int i(0); i<x_v.size(); i++) {
		x_v[i]=x[i];
	}
}

void algo_rohsa::minimize_spec_save(parameters &M, long n, long m, std::vector<double> &x_v, std::vector<double> &lb_v, int n_gauss_i, std::vector<double> &ub_v, std::vector<double> &line_v) {
/* Minimize_spec */ 
//int MAIN__(void)
    std::vector<double> _residual_(dim_v,0.);
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
			int nu = k+1;
			model[k]+= model_function(nu, params[0+3*i], params[1+3*i], params[2+3*i]);
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
			int nu = k+1;
			model[k]+= model_function(k+1, params[3*i], params[1+3*i], params[2+3*i]);
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
			double nu = double(k+1);
			dF_over_dB[0+3*i][k] += exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

			dF_over_dB[1+3*i][k] +=  params[3*i]*( nu - params[1+3*i])/pow(params[2+3*i],2.) * exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

			dF_over_dB[2+3*i][k] += params[3*i]*pow( nu - params[1+3*i] , 2.)/(pow(params[2+3*i],3.)) * exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

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
	for(int i=1; i<=M.n_gauss; i++) {
		std::vector<double> model_tab(dim_v,0.);
		std::vector<double> residual(dim_v,0.);
		std::vector<double> lb(3*i,0.);
		std::vector<double> ub(3*i,0.);

		init_bounds(M, line,i,lb,ub, true); //we consider {bool _init = true;} since we want to initialize the boundaries

		for(int j=0; j<i; j++) {

			for(int k=0; k<dim_v; k++) {			
				model_tab[k] += model_function(k+1,params[3*j], params[1+3*j], params[2+3*j]);
			}
		}
		
		for(int p(0); p<dim_v; p++) {	
			residual[p]=model_tab[p]-line[p];
		}

		std::vector<double> x(3*i,0.);

		for(int p(0); p<3*(i-1); p++){	
			x[p]=params[p];
		}
/*
		for(int c=0;c<x.size();c++)
		{
			std::cout<<"x["<<c<<"] = "<<x[c]<<std::endl;
		}
*/
		double argmin_res = minloc(residual);
		x[0+3*(i-1)] = line[int(argmin_res)]*M.amp_fact_init;
		x[1+3*(i-1)] = argmin_res+1;
		x[2+3*(i-1)] = M.sig_init;

/*
		printf("argmin_res = %f\n",argmin_res);
		printf("RESULTS (i = %d): \n",i);
		for(int p(0); p<3*i; p++){	
			printf("x[%d] = %f \n", p, x[p]);
		}
*/
		minimize_spec(M, 3*i, M.m, x, lb, i, ub, line);

/*
		printf("m = %d\n",M.m);
		for(int c=0;c<x.size();c++)
		{
			std::cout<<"x["<<c<<"] = "<<x[c]<<std::endl;
		}
//		std::cin.ignore();
		printf("------------------------------------------\n");
*/

		for(int p(0); p<3*(i); p++){	
			params[p]=x[p];
		}
	}

/*
		printf("FIN :\n");
		for(int p(0); p<3*M.n_gauss; p++){	
			printf("params[%d] = %f \n", p, params[p]);
		}
*/

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

void algo_rohsa::convolution_2D_mirror_flat(const parameters &M, double* image, double* &conv, int dim_y, int dim_x, int dim_k, float temps_transfert, float temps_mirroirs)
{
	int ii(0),jj(0),mm(0),nn(0),kCenterY(0), kCenterX(0);

	std::vector <std::vector<double>> ext_conv(dim_y+4, std::vector<double>(dim_x+4,0.));
	std::vector <std::vector<double>> extended(dim_y+4, std::vector<double>(dim_x+4,0.));

	float temps_temp = omp_get_wtime();

/*
	if(dim_x >=128){
	printf("print image :\n");
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<dim_x+4; j++)
		{
			printf("image[%d] = %.16f\n",j+(dim_x)*i,image[j+(dim_x)*i]);
		}
	}
    std::cin.ignore();
	}
*/

	for(int j(0); j<dim_y; j++)
	{
		for(int i(0); i<dim_x; i++)
		{
			extended[2+j][2+i]=image[dim_x*j+i];
		}
	}

/*
	if(dim_x >=128){
	printf("print ext image avant mirroir :\n");
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<dim_x+4; j++)
		{
			printf("extended[%d] = %.16f\n",j+(dim_x+4)*i,extended[i][j]);
		}
	}
	std::cin.ignore();
	}
*/

	for(int j(0); j<2; j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			extended[2+i][j] = image[dim_x*i+j];
		}
	}

	for(int i(0); i<2; i++)
	{
		for(int j(0); j<dim_x; j++)
		{
			extended[i][2+j] = image[dim_x*i+j];
		}
	}

	for(int j=dim_x; j<dim_x+2; j++)
	{
		for(int i=0; i<dim_y; i++)
		{
			extended[2+i][2+j]=image[dim_x*(i)+j-2];
		}
	}

	for(int j(0); j<dim_x; j++)
	{
		for(int i(dim_y); i<dim_y+2; i++)
		{
			extended[2+i][2+j]=image[dim_x*(i-2)+j];
		}
	}
	temps_mirroirs += omp_get_wtime() - temps_temp;

/*
	if(dim_x >=128){
	printf("print ext image :\n");
	for(int i=0; i<5; i++)
	{
		for(int j=0; j<dim_x+4; j++)
		{
			printf("extended[%d] = %.16f\n",j+(dim_x+4)*i,extended[i][j]);
		}
	}
	exit(0);
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
			conv[dim_x*i+j] = ext_conv[2+i][2+j];
		}
	}
}

void algo_rohsa::convolution_2D_mirror_flat(const parameters &M, float* image, float* &conv, int dim_y, int dim_x, int dim_k, float temps_transfert, float temps_mirroirs)
{
	int ii(0),jj(0),mm(0),nn(0),kCenterY(0), kCenterX(0);

	std::vector <std::vector<double>> ext_conv(dim_y+4, std::vector<double>(dim_x+4,0.));
	std::vector <std::vector<double>> extended(dim_y+4, std::vector<double>(dim_x+4,0.));

	for(int i(0); i<dim_y; i++)
	{
		for(int j(0); j<dim_x; j++)
		{
			extended[2+i][2+j]=image[i*dim_x+j];
		}
	}

	for(int i(0); i<dim_y; i++)
	{
		for(int j(0); j<2; j++)
		{
			extended[2+i][j] = image[i*dim_x+j];
		}
	}

	for(int i(0); i<2; i++)
	{
		for(int j(0); j<dim_x; j++)
		{
			extended[i][2+j] = image[i*dim_x+j];
		}
	}

	for(int i=0; i<dim_y; i++)
	{
		for(int j=dim_x; j<dim_x+2; j++)
		{
			extended[2+i][2+j]=image[i*dim_x+j-2];
		}
	}

	for(int i(dim_y); i<dim_y+2; i++)
	{
		for(int j(0); j<dim_x; j++)
		{
			extended[2+i][2+j]=image[(i-2)*dim_x+j];
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

	for(int i(0); i<dim_y; i++)
	{
		for(int j(0);j<dim_x;j++)
		{
			conv[i*dim_x+j] = ext_conv[2+i][2+j];
		}
	}
}


void algo_rohsa::convolution_2D_mirror(const parameters &M, const std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &conv, int dim_y, int dim_x, int dim_k)
{

	int ii(0),jj(0),mm(0),nn(0),kCenterY(0), kCenterX(0);

	std::vector <std::vector<double>> ext_conv(dim_y+4, std::vector<double>(dim_x+4,0.));
	std::vector <std::vector<double>> extended(dim_y+4, std::vector<double>(dim_x+4,0.));

	for(int i(0); i<dim_y; i++)
	{
		for(int j(0); j<dim_x; j++)
		{
			extended[2+i][2+j]=image[i][j];
		}
	}

	for(int i(0); i<dim_y; i++)
	{
		for(int j(0); j<2; j++)
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

	for(int i=0; i<dim_y; i++)
	{
		for(int j=dim_x; j<dim_x+2; j++)
		{
			extended[2+i][2+j]=image[i][j-2];
		}
	}

	for(int i(dim_y); i<dim_y+2; i++)
	{
		for(int j(0); j<dim_x; j++)
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

	for(int i(0); i<dim_y; i++)
	{
		for(int j(0);j<dim_x;j++)
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
	std::cout << "dim_v : " << dim_v <<  std::endl;
	std::cout << "dim_y : " << dim_y <<  std::endl;
	std::cout << "dim_x : " << dim_x <<  std::endl;

	std::cout << "vector.size() : " << vector.size() <<  std::endl;
	std::cout << "cube.size() : " << cube_3D.size() << " , " << cube_3D[0].size() << " , " << cube_3D[0][0].size() <<  std::endl;

//	std::cout << "avant cube[0][0][0] " <<  std::endl;

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

void algo_rohsa::initialize_array(double* array, int size, double value)
{
    for(int k=0; k<size; k++)
	{
		array[k]=value;
    }
}

void algo_rohsa::three_D_to_one_D(const std::vector<std::vector<std::vector<double>>> &cube_3D, std::vector<double> &vector, int dim_x, int dim_y, int dim_v)
{
	int k,j;
	#pragma omp parallel private(k,j) shared(vector,cube_3D,dim_x,dim_y,dim_v)
	{
	#pragma omp for
    for(k=0; k<dim_x; k++)
	    {
        for(j=0; j<dim_y; j++)
	        {
			for(int i(0); i<dim_v; i++)
				{
	            vector[k*dim_y*dim_v+j*dim_v+i] = cube_3D[i][j][k];
				}
			}
	    }
	}
}

void algo_rohsa::three_D_to_one_D(const std::vector<std::vector<std::vector<double>>> &cube_3D, double* vector, int dim_x, int dim_y, int dim_v)
{
	int k,j;
	#pragma omp parallel private(k,j) shared(vector,cube_3D,dim_x,dim_y,dim_v)
	{
	#pragma omp for
    for(k=0; k<dim_x; k++)
	    {
        for(j=0; j<dim_y; j++)
	        {
			for(int i(0); i<dim_v; i++)
				{
	            vector[k*dim_y*dim_v+j*dim_v+i] = cube_3D[i][j][k];
				}
			}
	    }
	}
}


void algo_rohsa::one_D_to_three_D_same_dimensions(double* vector, std::vector<std::vector<std::vector<double>>> &cube_3D, int dim_x, int dim_y, int dim_v)
{
	int k,j;
	#pragma omp parallel private(k,j) shared(vector,cube_3D,dim_x,dim_y,dim_v)
	{
	#pragma omp for
    for(k=0; k<dim_x; k++)
	    {
        for(j=0; j<dim_y; j++)
	        {
			for(int i(0); i<dim_v; i++)
				{
	            cube_3D[k][j][i] = vector[k*dim_y*dim_v+j*dim_v+i];
				}
			}
	    }
	}
}

/*
void algo_rohsa::three_D_to_one_D_same_dimensions_omp(const std::vector<std::vector<std::vector<double>>> &cube_3D, double* vector, int dim_x, int dim_y, int dim_v)
{
	#pragma omp parallel for private(k) shared(params,deriv,std_map,residual,indice_v,indice_y,indice_x)
    for(int k(0); k<dim_x; k++)
	    {
        for(int j(0); j<dim_y; j++)
	        {
			for(int i(0); i<dim_v; i++)
				{
	            vector[k*dim_y*dim_v+j*dim_v+i] = cube_3D[k][j][i];
				}
			}
	    }
}
*/
void algo_rohsa::three_D_to_one_D_same_dimensions(const std::vector<std::vector<std::vector<double>>> &cube_3D, double* vector, int dim_x, int dim_y, int dim_v)
{
	int k,j;
	#pragma omp parallel private(k,j) shared(vector,cube_3D,dim_x,dim_y,dim_v)
	{
	#pragma omp for
    for(k=0; k<dim_x; k++)
	    {
        for(j=0; j<dim_y; j++)
	        {
			for(int i(0); i<dim_v; i++)
				{
	            vector[k*dim_y*dim_v+j*dim_v+i] = cube_3D[k][j][i];
				}
			}
	    }
	}
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


template void algo_rohsa::descente<double>(parameters&, std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<std::vector<double>>>&);
template void algo_rohsa::reshape_down<double>(std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<std::vector<double>>>&);
template void algo_rohsa::set_stdmap<double>(std::vector<std::vector<double>>&, std::vector<std::vector<std::vector<double>>>&, int, int);
template void algo_rohsa::set_stdmap_transpose<double>(std::vector<std::vector<double>>&, std::vector<std::vector<std::vector<double>>>&, int, int);
template void algo_rohsa::update_clean<double>(parameters&, std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<double>>&, int, int, int, std::vector<double>&);

template void algo_rohsa::minimize_clean<double>(parameters&, long, long, double*, double*, double*, std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<double>>&, int, int, int, double*);
template void algo_rohsa::minimize_clean_cpu<double>(parameters&, long, long, double*, double*, double*, std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<double>>&, int, int, int, double*);
template void algo_rohsa::minimize_clean_gpu<double>(parameters&, long, long, double*, double*, double*, std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<double>>&, int, int, int, double*);

template void algo_rohsa::f_g_cube_fast<double>(parameters&, double&, double*, int, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, std::vector<std::vector<double>>&);
template void algo_rohsa::f_g_cube_fast_clean<double>(parameters&, double&, double*, int, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, std::vector<std::vector<double>>&);
template void algo_rohsa::f_g_cube_fast_clean_optim_CPU<double>(parameters&, double&, double*, int, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, std::vector<std::vector<double>>&);
template void algo_rohsa::f_g_cube_fast_clean_optim_CPU_lib<double>(parameters&, double&, double*, int, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, std::vector<std::vector<double>>&, double** assist_buffer);
template void algo_rohsa::f_g_cube_fast_without_regul<double>(parameters&, double&, double*, int, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, std::vector<std::vector<double>>&);
template void algo_rohsa::f_g_cube_vector<double>(parameters&, double&, double*, int, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, std::vector<std::vector<double>>&);

template void algo_rohsa::f_g_cube_cuda_L<double>(parameters&, double&, double*, int, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, std::vector<std::vector<double>>&, double*);
template void algo_rohsa::f_g_cube_cuda_L_clean<double>(parameters&, double&, double*, int, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, std::vector<std::vector<double>>&, double*);
template void algo_rohsa::f_g_cube_cuda_L_clean_lib(parameters&, double&, double*, int, double*, int, int, int, std::vector<std::vector<double>>&, double*);
template void algo_rohsa::f_g_cube_cuda_L_clean_lib_device_fg(parameters&, double&, double*, int, double*, int, int, int, std::vector<std::vector<double>>&, double*);
template void algo_rohsa::f_g_cube_omp_without_regul<double>(parameters&, double&, double*, int, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, std::vector<std::vector<double>>&);

//template <typename T> void algo_rohsa::set_stdmap_transpose(std::vector<std::vector<T>> &std_map, std::vector<std::vector<std::vector<T>>> &cube, int lb, int ub){