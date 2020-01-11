#include "algo_rohsa.h"
#include <omp.h>
#include <array>


algo_rohsa::algo_rohsa(model &M, hypercube &Hypercube)
{

	this->file = Hypercube; //The hypercube is not modified then
	this->dim_cube = Hypercube.get_dim_cube();
	this->dim_x = dim_cube[0];
	this->dim_y = dim_cube[1];
	this->dim_v = dim_cube[2];
	
	std_spectrum(dim_x, dim_y, dim_v); //oublier
	mean_spectrum(dim_x, dim_y, dim_v);
	max_spectrum(dim_x, dim_y, dim_v); //oublier


	double max_mean_spect = *std::max_element(mean_spect.begin(), mean_spect.end());

	max_spectrum_norm(this->dim_x,this->dim_y, this->dim_v, max_mean_spect);

	std::cout << " descent : "<< M.descent << std::endl;

	std::vector<std::vector<std::vector<double>>> grid_params, fit_params;

// can't define the proper variable in the loop 
	if(M.descent){
	std::vector<std::vector<std::vector<double>>> grid_params_(3*(M.n_gauss+(file.nside*M.n_gauss_add)), std::vector<std::vector<double>>(dim_y, std::vector<double>(dim_x,0.)));
	std::vector<std::vector<std::vector<double>>> fit_params_(3*(M.n_gauss+(file.nside*M.n_gauss_add)), std::vector<std::vector<double>>(1, std::vector<double>(1,0.)));
	grid_params=grid_params_;
	fit_params=fit_params_;
	}
	else{
	std::vector<std::vector<std::vector<double>>> grid_params_(3*(M.n_gauss+M.n_gauss_add), std::vector<std::vector<double>>(dim_y, std::vector<double>(dim_x,0.)));
	grid_params=grid_params_;
	}

	std::vector<double> b_params(M.n_gauss,0.);


/*		M.fit_params = fit_params;

		for(int p=0; p<M.fit_params.size();p++){
			std::cout << "fit_params : "<<M.fit_params[p]<<  std::endl;
                        }
*/				
//		std::cout << "test fit_params : "<<fit_params[0][0][0]<<std::endl;
		
	if(M.descent)
	{
		descente(M, grid_params, fit_params);
	}
/*
	std::cout << "params.size() : " << fit_params.size() << " , " << fit_params[0].size() << " , " << fit_params[0][0].size() <<  std::endl;

	for(int i(0); i<fit_params.size(); i++) {
		for(int j(0); j<fit_params[0].size(); j++) {
			for(int k(0); k<fit_params[0][0].size(); k++) {
				std::cout<<"Après setulb, params["<<i<<"]["<<j<<"]["<<k<<"] = "<<fit_params[i][j][k]<<std::endl;
			}
		}
	}
*/
}

void algo_rohsa::descente(model &M, std::vector<std::vector<std::vector<double>>> &grid_params, std::vector<std::vector<std::vector<double>>> &fit_params){

	temps_f_g_cube = 0.; 
	temps_conv = 0.;
	temps_deriv = 0.;
	temps_tableaux = 0.;
	temps_bfgs = 0.;
	temps_update_beginning = 0.;

	for(int i=0;i<M.n_gauss; i++){
		fit_params[0+3*i][0][0] = 0.;
		fit_params[1+3*i][0][0] = 0.;
		fit_params[2+3*i][0][0] = 0.;
	}
		std::cout << "fit_params.size() : " << fit_params.size() << " , " << fit_params[0].size() << " , " << fit_params[0][0].size() <<  std::endl;

	std::cout << "grid_params.size() : "<< grid_params.size() << " , " << grid_params[0].size()  << " , " << grid_params[0][0].size() << std::endl;

		double temps1_descente = omp_get_wtime();

		std::vector<double> fit_params_flat(fit_params.size(),0.); //used below

		double temps_upgrade=0.;
		double temps_multiresol=0.;
		double temps_init = 0.;
		double temps_mean_array=0.;
		int n;
//		#pragma omp parallel private(n) shared(temps_upgrade, temps_multiresol, temps_init, temps_mean_array,M, fit_params_flat,file)
//		{
//		#pragma omp for 
		for(n=0; n<file.nside; n++)
		{
			int power(pow(2,n));

			std::cout << " power = " << power << std::endl;

			std::vector<std::vector<std::vector<double>>> cube_mean(power, std::vector<std::vector<double>>(power,std::vector<double>(dim_v,1.)));

			double temps1_mean_array = omp_get_wtime();
			mean_array(power, cube_mean);
			double temps2_mean_array = omp_get_wtime();
			temps_mean_array+=temps2_mean_array-temps1_mean_array;

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
				double temps1_init = omp_get_wtime();
				init_spectrum(M, cube_mean_flat, fit_params_flat);

//				init_spectrum(M, cube_mean_flat, std_spect); //option spectre
//				init_spectrum(M, cube_mean_flat, max_spect); //option max spectre
//				init_spectrum(M, cube_mean_flat, max_spect_norm); //option norme spectre
				for(int e(0); e<fit_params_flat.size(); e++) {
					fit_params[e][0][0] = fit_params_flat[e]; //cache
					}

				double temps2_init = omp_get_wtime();
				temps_init += temps2_init - temps1_init;

//				for(int i(0); i<M.n_gauss; i++) {
//					b_params[i]= fit_params_flat[2+3*i];
//					}
				}

			double temps1_upgrade = omp_get_wtime();
			if(M.regul==false) {
				for(int e(0); e<fit_params.size(); e++) {
					fit_params[e][0][0]=fit_params_flat[e];
					grid_params[e][0][0] = fit_params[e][0][0];
					}

				upgrade(M ,cube_mean, fit_params, power);
			} else if(M.regul) {
				if (n==0){
					upgrade(M ,cube_mean, fit_params, power);
				}
				if (n>0 and n<file.nside){
					std::vector<std::vector<double>> std_map(power, std::vector<double>(power,0.));
					if (M.noise){
						
					}
					else if (M.noise==false){
						set_stdmap_transpose(std_map, cube_mean, M.lstd, M.ustd);
					}

					update(M, cube_mean, fit_params, std_map, power, power, dim_v);
					if (M.n_gauss_add != 0){
						
					//Add new Gaussian if one reduced chi square > 1  
					}
				}
			}

			if (M.save_grid){

				//save grid in file
			}



			double temps2_upgrade = omp_get_wtime();
			temps_upgrade+=temps2_upgrade-temps1_upgrade;



			go_up_level(fit_params);
			M.fit_params = fit_params; //updating the model class
		}

	std::vector<std::vector<double>> std_map(file.dim_data[0], std::vector<double>(file.dim_data[1],0.));

	if(M.noise){


	} else {

		set_stdmap(std_map, this->file.cube, M.lstd, M.ustd);
	}

	if(M.regul){
		std::cout<<"Update last level"<<std::endl;
		update(M, this->file.cube, fit_params, std_map, this->dim_x, this->dim_y, this->dim_v);
	}

	reshape_down(fit_params, grid_params);
	M.grid_params = grid_params;


























	for(int i(0); i<fit_params.size(); i++) {
		for(int j(0); j<fit_params[0].size(); j++) {
			for(int k(0); k<fit_params[0][0].size(); k++) {
				std::cout<<"Après setulb, fit_params["<<i<<"]["<<j<<"]["<<k<<"] = "<<fit_params[i][j][k]<<std::endl;
			}
		}
	}


		double temps2_descente = omp_get_wtime();
//		std::cout<<"fit_params_flat["<<0<<"]= "<<"  vérif:  "<<fit_params[0][0][0]<<std::endl;

		std::cout<<"Temps TOTAL de descente : "<<temps2_descente - temps1_descente <<std::endl;
		std::cout<<"Temps de upgrade : "<< temps_upgrade <<std::endl;
		std::cout<<"Temps de mean_array : "<<temps_mean_array<<std::endl;
		std::cout<<"Temps de init : "<<temps_init<<std::endl;
	std::cout<< "temps d'exécution dF/dB : "<<temps_f_g_cube<<std::endl;
	std::cout<< "Temps d'exécution convolution : " << temps_conv <<std::endl;
	std::cout<< "Temps d'exécution deriv : " << temps_deriv  <<std::endl;
	std::cout<< "Temps d'exécution tableaux : " << temps_tableaux <<std::endl;


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
				tab2[i][k][j] = tab1[i][offset_w+j][offset_h+k];
			}
		}
	}

}

void algo_rohsa::update(model &M, std::vector<std::vector<std::vector<double>>> &cube, std::vector<std::vector<std::vector<double>>> &params, std::vector<std::vector<double>> &std_map, int indice_x, int indice_y, int indice_v) {


	int n_beta = 3*M.n_gauss * indice_y * indice_x;

	std::vector<double> lb(n_beta);
	std::vector<double> ub(n_beta);
	std::vector<double> beta(n_beta);

	std::vector<std::vector<std::vector<double>>> lb_3D(3*M.n_gauss, std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
	std::vector<std::vector<std::vector<double>>> ub_3D(3*M.n_gauss, std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));

	std::vector<double> mean_amp(M.n_gauss,0.);
	std::vector<double> mean_mu(M.n_gauss,0.);
	std::vector<double> mean_sig(M.n_gauss,0.);

	std::vector<std::vector<double>> image_amp(indice_y, std::vector<double>(indice_x,0.));
	std::vector<std::vector<double>> image_mu(indice_y, std::vector<double>(indice_x,0.));
	std::vector<std::vector<double>> image_sig(indice_y, std::vector<double>(indice_x,0.));

	std::vector<double> ravel_amp(indice_y*indice_x,0.);
	std::vector<double> ravel_mu(indice_y*indice_x,0.);
	std::vector<double> ravel_sig(indice_y*indice_x,0.);

	std::vector<double> cube_flat(cube[0][0].size(),0.);
	std::vector<double> lb_3D_flat(lb_3D.size(),0.);
	std::vector<double> ub_3D_flat(ub_3D.size(),0.);

	for(int j=0; j<indice_x; j++) {
		for(int i=0; i<indice_y; i++) {
			for(int p=0; p<cube_flat.size(); p++){
				cube_flat[p]=cube[i][j][p];
			}
			for(int p=0; p<3*M.n_gauss; p++){
				lb_3D_flat[p]=lb_3D[p][i][j];
				ub_3D_flat[p]=ub_3D[p][i][j];
			}
			init_bounds(M, cube_flat, M.n_gauss, lb_3D_flat, ub_3D_flat);
			for(int p=0; p<3*M.n_gauss; p++){
				lb_3D[p][i][j]=lb_3D_flat[p];
				ub_3D[p][i][j]=ub_3D_flat[p];
			}
		}
	}

	ravel_3D(lb_3D, lb, 3*M.n_gauss, indice_y, indice_x);
	ravel_3D(ub_3D, ub, 3*M.n_gauss, indice_y, indice_x);
	ravel_3D(params, beta, 3*M.n_gauss, indice_y, indice_x);

	for(int i=0; i<M.n_gauss; i++){
		for(int j=0; j<params[0].size(); j++){
			for(int k=0; k<params[0][0].size(); k++){
				image_amp[j][k]=params[3*i][j][k];
				image_mu[j][k]=params[1+3*i][j][k];
				image_sig[j][k]=params[2+3*i][j][k];
			}
		}

		ravel_2D(image_amp, ravel_amp, indice_y, indice_x);
		ravel_2D(image_mu, ravel_mu, indice_y, indice_x);
		ravel_2D(image_sig, ravel_sig, indice_y, indice_x);

		mean_amp[i]=mean(ravel_amp);
		mean_mu[i]=mean(ravel_mu);
		mean_sig[i]=mean(ravel_sig);
	}


	minimize(M, n_beta, M.m, beta, lb, ub, cube, std_map, mean_amp, mean_mu, mean_sig, indice_x, indice_y, indice_v); 

	unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);


}

void algo_rohsa::set_stdmap(std::vector<std::vector<double>> &std_map, std::vector<std::vector<std::vector<double>>> &cube, int lb, int ub){
	std::vector<double> line(ub-lb+1,0.);
	int dim[3];
	dim[2]=cube[0][0].size();
	dim[1]=cube[0].size();
	dim[0]=cube.size();
	for(int j=0; j<dim[1]; j++){
		for(int i=0; i<dim[0]; i++){
			for(int p=0; p<line.size(); p++){
				line[p] = cube[i][j][p+lb];
			}
			std_map[i][j] = Std(line);
		}
	}
}

void algo_rohsa::set_stdmap_transpose(std::vector<std::vector<double>> &std_map, std::vector<std::vector<std::vector<double>>> &cube, int lb, int ub){
	std::vector<double> line(ub-lb+1,0.);
	int dim[3];
	dim[2]=cube[0][0].size();
	dim[1]=cube[0].size();
	dim[0]=cube.size();
	for(int j=0; j<dim[1]; j++){
		for(int i=0; i<dim[0]; i++){
			for(int p=0; p<line.size(); p++){
				line[p] = cube[i][j][p+lb];
			}
			std_map[j][i] = Std(line);
		}
	}
}
void algo_rohsa::f_g_cube_fast(model &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig, double mesure_temps){

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
		convolution_2D_mirror(M, image_mu, conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror(M, image_sig, conv_amp, indice_y, indice_x,3);
	
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
					deriv[0+3*i][j][l] += exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
	
					deriv[1+3*i][j][l] +=  params[3*i][j][l]*( double(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
	
					deriv[2+3*i][j][l] += params[3*i][j][l]*pow( double(k+1) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*(residual[l][j][k]/pow(std_map[j][l],2));
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
void algo_rohsa::f_g_cube(model &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig){

std::vector<std::vector<std::vector<double>>> dR_over_dB(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
std::vector<std::vector<std::vector<std::vector<double>>>> dF_over_dB(3*M.n_gauss,std::vector<std::vector<std::vector<double>>>(indice_v,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.))));
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

int n_beta = (3*M.n_gauss*indice_x*indice_y);//+M.n_gauss;

for(int i = 0; i< n; i++){
	g[i]=0.;
}
f=0.;

double temps1_tableaux = omp_get_wtime();

unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

/*for(int i = 0; i<M.n_gauss; i++){
	b_params[i]=beta[n_beta-M.n_gauss+i];
}*/
//cout.precision(dbl::max_digits10);

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

int i,k,j,l;

//nouveau code
#pragma omp parallel private(i,k,j) shared(dF_over_dB,params,M,indice_v,indice_y,indice_x)
{
#pragma omp for
for(int i=0; i<M.n_gauss; i++){
	for(k=0; k<indice_v; k++){
		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){
				dF_over_dB[0+3*i][k][j][l] += exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

				dF_over_dB[1+3*i][k][j][l] +=  params[3*i][j][l]*( double(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );

				dF_over_dB[2+3*i][k][j][l] += params[3*i][j][l]*pow( double(k+1) - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( double(k+1)-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) );
			}
		}
	}
}
}

double temps2_dF_dB = omp_get_wtime();

double temps1_deriv = omp_get_wtime();
#pragma omp parallel private(k,l) shared(dF_over_dB,params,M,indice_v,indice_y,indice_x)
{
#pragma omp for
for(k=0; k<indice_v; k++){
	for(l=0; l<3*M.n_gauss; l++){
		for(i=0; i<indice_y; i++){
			for(j=0; j<indice_x; j++){
				if(std_map[i][j]>0.){
					deriv[l][i][j]+= dF_over_dB[l][k][i][j]*residual[j][i][k]/pow(std_map[i][j],2);
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
	convolution_2D_mirror(M, image_mu, conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, image_sig, conv_amp, indice_y, indice_x,3);

	convolution_2D_mirror(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
	convolution_2D_mirror(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);



	for(int j=0; j<indice_x; j++){
		for(int i=0; i<indice_y; i++){
			f+= 0.5*M.lambda_amp*pow(conv_amp[i][j],2) + 0.5*M.lambda_var_amp*pow(image_amp[i][j]-mean_amp[k],2);
			f+= 0.5*M.lambda_mu*pow(conv_mu[i][j],2) + 0.5*M.lambda_var_mu*pow(image_mu[i][j]-mean_mu[k],2);
			f+= 0.5*M.lambda_sig*pow(conv_sig[i][j],2) + 0.5*M.lambda_var_sig*pow(image_sig[i][j]-mean_sig[k],2);

			dR_over_dB[0+3*k][i][j] = M.lambda_amp*conv_conv_amp[i][j]+M.lambda_var_amp*(image_amp[i][j]-mean_amp[k]);
			dR_over_dB[1+3*k][i][j] = M.lambda_mu*conv_conv_mu[i][j]+M.lambda_var_mu*(image_mu[i][j]-mean_mu[k]);
			dR_over_dB[2+3*k][i][j] = M.lambda_sig*conv_conv_sig[i][j]+M.lambda_var_sig*(image_sig[i][j]-mean_sig[k]);
		}
	}

} 

	for(int j=0; j<indice_x; j++){
		for(int i=0; i<indice_y; i++){
			for(int l=0; l<3*M.n_gauss; l++){
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
	
void algo_rohsa::minimize(model &M, long n, long m, std::vector<double> &x_v, std::vector<double> &lb_v, std::vector<double> &ub_v, std::vector<std::vector<std::vector<double>>> &cube, std::vector<std::vector<double>> &std_map, std::vector<double> &mean_amp, std::vector<double> &mean_mu, std::vector<double> &mean_sig, int indice_x, int indice_y, int indice_v) {
//int MAIN__(void)
    /* System generated locals */
    int i__1;
    double d__1, d__2;
    /* Local variables */
    double f, g[n];
    int i__;

    int taille_wa = 2*M.m*n+5*n+11*M.m*M.m+8*M.m;
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
    double x[x_v.size()];
    double lb[lb_v.size()];
    double ub[ub_v.size()];

    for(int i(0); i<x_v.size(); i++) {
	x[i]=x_v[i];
    } 
    for(int i(0); i<lb_v.size(); i++) {
	lb[i]=lb_v[i];
    } 
    for(int i(0); i<ub_v.size(); i++) {
	ub[i]=ub_v[i];
    } 

    for(int i(0); i<n; i++) {
	g[i]=0.;
    } 

     f=0.;

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

	double temps1_f_g_cube = omp_get_wtime();

    while(IS_FG(*task) or *task==NEW_X or *task==START){ 
    /*     This is the call to the L-BFGS-B code. */
//    std::cout<<" Début appel BFGS "<<std::endl;

    setulb(&n, &m, x, lb, ub, nbd, &f, g, &factr, &pgtol, wa, iwa, task, 
            &M.iprint, csave, lsave, isave, dsave);

/*     if (s_cmp(task, "FG", (ftnlen)2, (ftnlen)2) == 0) { */
    if ( IS_FG(*task) ) {

	f_g_cube(M, f, g, n,cube, x, indice_v, indice_y, indice_x, std_map, mean_amp, mean_mu, mean_sig);
//	f_g_cube_fast(M, f, g, n,cube, x, indice_v, indice_y, indice_x, std_map, mean_amp, mean_mu, mean_sig);

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

	double temps2_f_g_cube = omp_get_wtime();

	std::cout<< "Temps de calcul gradient : " << temps2_f_g_cube - temps1_f_g_cube<<std::endl;
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

void algo_rohsa::upgrade(model &M, std::vector<std::vector<std::vector<double>>> &cube, std::vector<std::vector<std::vector<double>>> &params, int power) {
        int i,j;
//        int nb_threads = omp_get_max_threads();
//        printf(">> omp_get_max_thread()\n>> %i\n", nb_threads);

//      #pragma omp parallel shared(cube,params) shared(power) shared(M)
//      {
        std::vector<double> line(dim_v,0.);
        std::vector<double> x(3*M.n_gauss,0.);
        std::vector<double> lb(3*M.n_gauss,0.);
        std::vector<double> ub(3*M.n_gauss,0.);
//        printf("thread:%d\n", omp_get_thread_num());
//      #pragma omp for private(i,j)
        for(i=0;i<power; i++){
                for(j=0;j<power; j++){

                        int p;
                        for(p=0; p<cube[0][0].size();p++){

                                line[p]=cube[i][j][p];
                        }
                        for(p=0; p<params.size(); p++){
                                x[p]=params[p][i][j]; //cache
                        }
                        init_bounds(M, line, M.n_gauss, lb, ub);
                        minimize_spec(M,3*M.n_gauss ,M.m ,x ,lb , M.n_gauss, ub ,line);
                        for(p=0; p<params.size();p++){
                                params[p][i][j]=x[p]; //cache
//                              std::cout << "p = "<<p<<  std::endl;
                        }
                }
//      }
        }
}



void algo_rohsa::init_bounds(model &M, std::vector<double> line, int n_gauss_local, std::vector<double> &lb, std::vector<double> &ub) {

	double max_line = *std::max_element(line.begin(), line.end());
//	std::cout<<"affiche "<<max_line<<std::endl;
	for(int i(0); i<n_gauss_local; i++) {
		lb[0+3*i]=0.;
		ub[0+3*i]=max_line;

		lb[1+3*i]=0.;
		ub[1+3*i]=dim_v;

		lb[2+3*i]=M.lb_sig;
		ub[2+3*i]=M.ub_sig;
	}
}

double algo_rohsa::model_function(int x, double a, double m, double s) {

	return a*exp(-pow((double(x)-m),2.) / (2.*pow(s,2.)));

}

int algo_rohsa::minloc(std::vector<double> &tab) {
	return std::distance(tab.begin(), std::min_element( tab.begin()+1, tab.end() ));
}

void algo_rohsa::minimize_spec(model &M, long n, long m, std::vector<double> &x_v, std::vector<double> &lb_v, int n_gauss_i, std::vector<double> &ub_v, std::vector<double> &line_v) {
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

/*     if (s_cmp(task, "FG", (ftnlen)2, (ftnlen)2) == 0) { */
    if ( IS_FG(*task) ) {

	myresidual(x, line, _residual_, n_gauss_i);
	f = myfunc_spec(_residual_);
	mygrad_spec(g, _residual_, x, n_gauss_i);

	}

/*
    if (*task==NEW_X ) {
	if (isave[33] >= M.maxiter) {
		*task = STOP_ITER;
		}
	
	if (dsave[12] <= (fabs(f) + 1.) * 1e-10) {
		*task = STOP_GRAD;
	}
     }
*/

        /*          go back to the minimization routine. */
//if (compteurX<100000000){        
//	goto L111;
//}

	}

	for(int i(0); i<x_v.size(); i++) {
		x_v[i]=x[i];
	}

}

double algo_rohsa::myfunc_spec(std::vector<double> &residual) {
	double S(0.);
	for(int p(0); p<residual.size(); p++) {
		S+=pow(residual[p],2);
	}
	return 0.5*S;
}

void algo_rohsa::myresidual(double params[], double line[], std::vector<double> &residual, int n_gauss_i) {
	int k;
	std::vector<double> model(dim_v,0.);
	for(int i(0); i<n_gauss_i; i++) {
		for(k=1; k<=dim_v; k++) {
			model[k-1]+= model_function(k, params[3*i], params[1+3*i], params[2+3*i]);
		}
	}
	for(int p=0; p<residual.size(); p++) {
		residual[p]=model[p]-line[p]; 
	}
}

void algo_rohsa::myresidual(std::vector<double> &params, std::vector<double> &line, std::vector<double> &residual, int n_gauss_i) {
	int k;
	std::vector<double> model(dim_v,0.);

	for(int i(0); i<n_gauss_i; i++) {
		for(k=1; k<=dim_v; k++) {
			model[k-1]+= model_function(k, params[3*i], params[1+3*i], params[2+3*i]);
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

void algo_rohsa::init_spectrum(model &M, std::vector<double> &line, std::vector<double> &params) {

	std::vector<double> model_tab(dim_v,0.);
	std::vector<double> residual(dim_v,0.);
	int i;

	for(i=1; i<=M.n_gauss; i++) {
		std::vector<double> lb(3*i,0.);
		std::vector<double> ub(3*i,0.);
		int rang = std::distance(residual.begin(), std::min_element( residual.begin(), residual.end() ));

		init_bounds(M, line,i,lb,ub);

		for(int j(0); j<i; j++) {

			for(int k=0; k<dim_v; k++) {			
				model_tab[k]+= model_function(k+1,params[3*j], params[1+3*j], params[2+3*j]);
			}
		}
		
		for(int p(0); p<dim_v; p++) {	
			residual[p]=model_tab[p]-line[p];
		}
		
		std::vector<double> x(3*i,0.);

		for(int p(0); p<3*(i); p++){	
			x[p]=params[p];
		}
		
		x[1+3*(i-1)] = minloc(residual)+1;
		x[0+3*(i-1)] = line[int(x[1+3*(i-1)])-1]*M.amp_fact_init;
		x[2+3*(i-1)] = M.sig_init;

		minimize_spec(M, 3*i, M.m, x, lb, i, ub, line);

		for(int p(0); p<3*(i); p++) {
			params[p] = x[p];
		}
		for(int p(0); p<3*(i); p++) {
			params[p] = params[p];
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

void algo_rohsa::convolution_2D_mirror(model &M, std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &conv, int dim_y, int dim_x, int dim_k)
{
	int ii(0),jj(0),mm(0),nn(0),kCenterY(0), kCenterX(0);

	std::vector <std::vector<double>> ext_conv(dim_x+4, std::vector<double>(dim_y+4));
	std::vector <std::vector<double>> extended(dim_x+4, std::vector<double>(dim_y+4));

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

	for(int j(dim_x); j<dim_x+2; j++)
	{
		for(int i(0); i<dim_y; i++)
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
	int i__(0);

	for(int k(0); k<dim_x; k++)
	{
		for(int j(0);j<dim_y;j++)
		{
			vector[i__] = map[j][k];
			i__++;
		}
	}
}

// It transforms a 1D vector into a contiguous flattened 1D array from a 3D array, the interest is close to the valarray's one

void algo_rohsa::ravel_3D(const std::vector<std::vector<std::vector<double>>> &cube, std::vector<double> &vector, int dim_v, int dim_y, int dim_x)
{
        int i__(0);

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
}

void algo_rohsa::ravel_3D(const std::vector<std::vector<std::vector<double>>> &cube, double vector[], int dim_v, int dim_y, int dim_x)
{
        int i__(0);

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
}

void algo_rohsa::unravel_3D(double vector[], std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x)
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


