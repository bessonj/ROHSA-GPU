#include "f_g_cube.hpp"

template <typename T> 
void f_g_cube_fast_unidimensional(parameters<T> &M, T &f, T* g, int n, T* cube, std::vector<std::vector<std::vector<T>>>& cube_for_cache, T* beta, int indice_v, int indice_y, int indice_x, T* std_map, double* temps){

	double temps_conv;
	double temps_deriv;
	double temps_tableaux;
	double temps_copy;
	double temps_f_g_cube;

    T Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};

	std::vector<T> b_params(M.n_gauss,0.);
	int taille_params_flat[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_residual[] = {indice_v, indice_y, indice_x};
	int taille_std_map_[] = {indice_y, indice_x};
	int taille_beta[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_cube[] = {indice_v, indice_y, indice_x};
	int taille_image_conv[] = {indice_y, indice_x};
	int product_residual = taille_residual[0]*taille_residual[1]*taille_residual[2];
	int product_params_flat = taille_params_flat[0]*taille_params_flat[1]*taille_params_flat[2];
	int product_std_map_ = taille_std_map_[0]*taille_std_map_[1];
	int product_cube = taille_cube[0]*taille_cube[1]*taille_cube[2]; 
	int product_beta = taille_beta[0]*taille_beta[1]*taille_beta[2];
	int product_image_conv = taille_image_conv[0]*taille_image_conv[1];

	size_t size_res = product_residual * sizeof(T);
	size_t size_std = product_std_map_ * sizeof(T);
	size_t size_image_conv = product_image_conv * sizeof(T);

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
	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			std_map_[j*indice_y+i] = std_map[i*indice_x+j];
		}
	}
	for(int i = 0; i<n_beta; i++){
		g[i]=0.;
	}
	f=0.;
	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}
	temps_copy+=omp_get_wtime()-temps1_ravel;

	double temps1_tableaux = omp_get_wtime();
/*
	for(int i = 0; i<M.n_gauss; i++){
		printf("b_params[%d]= %.16f\n",i, b_params[i]);
	}
	for(int i = 0; i<n_beta; i++){
		printf("beta[%d]= %.16f\n",i, beta[i]);
	}
*/
	int i,j,l;


	T* hypercube_tilde = NULL;
	hypercube_tilde = (T*)malloc(indice_x*indice_y*indice_v*sizeof(T));

	T* par = NULL;
	par = (T*)malloc(3*sizeof(T));

//	omp_set_num_threads(2);
	#pragma omp parallel private(j,i) shared(hypercube_tilde, residual ,par,M,beta,std_map_,indice_v,indice_y,indice_x)
	{
	#pragma omp for
	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
			for (int p=0; p<indice_v; p++){
				T accu = 0.;
				for (int p_par=0; p_par<M.n_gauss; p_par++){
					par[0]= beta[j*3*M.n_gauss*indice_y+i*3*M.n_gauss+(3*p_par+0)];
					par[1]= beta[j*3*M.n_gauss*indice_y+i*3*M.n_gauss+(3*p_par+1)];
					par[2]= beta[j*3*M.n_gauss*indice_y+i*3*M.n_gauss+(3*p_par+2)];
					accu+= par[0]*exp(-powf(T(p+1)-par[1],2)/(2*powf(par[2],2.)));
				}
			residual[j*indice_y*indice_v+i*indice_v+p] = accu-cube_for_cache[j][i][p];
//				hypercube_tilde[j*indice_y*indice_v+i*indice_v+p] = accu;	
			}
		}
	}
	}
	free(hypercube_tilde);
	free(par);

	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
			if(std_map_[j*indice_y+i]>0.){
				T accu = 0.;
				for (int p=0; p<indice_v; p++){
					accu+=powf(residual[j*indice_y*indice_v+i*indice_v+p],2.);
				}
	//				printf("accu = %f\n",accu);
				f += 0.5*accu/powf(std_map_[j*indice_y+i],2.); //std_map est arrondie... 
			}
		}
	}
/*
	printf("M.lambda_var_sig = %.16f\n",M.lambda_var_sig);
	printf("M.lambda_var_sig = %.16f\n",M.lambda_var_sig);
	printf("M.lambda_amp = %.16f\n",M.lambda_amp);
	printf("M.lambda_mu = %.16f\n",M.lambda_mu);
	printf("M.lambda_sig = %.16f\n",M.lambda_sig);
	printf("M.n_gauss = %d\n",M.n_gauss);
	std::cin.ignore();
*/

	temps_tableaux+=omp_get_wtime()-temps1_tableaux;

	for(int i=0; i<M.n_gauss; i++){
		double temps1_conv = omp_get_wtime();
		for(int q=0; q<indice_x; q++){
			for(int p=0; p<indice_y; p++){
				image_amp[indice_x*p+q]= beta[q*indice_y*3*M.n_gauss + p*3*M.n_gauss+(0+3*i)];
				image_mu[indice_x*p+q]=beta[q*indice_y*3*M.n_gauss + p*3*M.n_gauss+(1+3*i)];
				image_sig[indice_x*p+q]=beta[q*indice_y*3*M.n_gauss + p*3*M.n_gauss+(2+3*i)];
			}
		}
/*
		for(int q=0; q<indice_x; q++){
			for(int p=0; p<indice_y; p++){
				printf("image_sig[%d] = %.16f\n",indice_x*p+q,image_sig[indice_x*p+q]);
			}
		}
*/
		convolution_2D_mirror_flat<T>(M, image_amp, conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror_flat<T>(M, image_mu, conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror_flat<T>(M, image_sig, conv_sig, indice_y, indice_x,3);
		convolution_2D_mirror_flat<T>(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror_flat<T>(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror_flat<T>(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);

		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){
				f+= 0.5*M.lambda_amp*powf(conv_amp[j*indice_x+l],2);
				f+= 0.5*M.lambda_mu*powf(conv_mu[j*indice_x+l],2);
				f+= 0.5*M.lambda_sig*powf(conv_sig[j*indice_x+l],2) + 0.5*M.lambda_var_sig*powf(image_sig[j*indice_x+l]-b_params[i],2);
//				printf("b_params[i] = %.16f\n",b_params[i]);
//				printf("image_sig[j*indice_x+l] = %.16f\n",image_sig[j*indice_x+l]);
				g[n_beta-M.n_gauss+i] += M.lambda_var_sig*(b_params[i]-image_sig[j*indice_x+l]);
				g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss+(0+3*i)] += M.lambda_amp*conv_conv_amp[j*indice_x+l];
				g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss+(1+3*i)] += M.lambda_mu*conv_conv_mu[j*indice_x+l];
				g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss+(2+3*i)] += M.lambda_sig*conv_conv_sig[j*indice_x+l]+M.lambda_var_sig*(image_sig[j*indice_x+l]-b_params[i]);
			}
		}
		temps_conv+=omp_get_wtime()-temps1_conv;

	}

/*
	for(int i = 0; i<n_beta; i++){
		printf("g[%d]= %.16f\n",i, g[i]);
	}
	std::cin.ignore();
*/

	double temps1_deriv = omp_get_wtime();
	#pragma omp parallel private(j,l) shared(g,M,beta,std_map_,residual,indice_v,indice_y,indice_x)
	{
	#pragma omp for
	for(l=0; l<indice_x; l++){
		for(j=0; j<indice_y; j++){
			if(std_map_[l*indice_y+j]>0.){
				for(int i=0; i<M.n_gauss; i++){
					T par0 = beta[l*indice_y*3*M.n_gauss +j*3*M.n_gauss +(0+3*i)];
					T par1_ = beta[l*indice_y*3*M.n_gauss +j*3*M.n_gauss +(1+3*i)];
					T par2 = beta[l*indice_y*3*M.n_gauss +j*3*M.n_gauss +(2+3*i)];							
					for(int k=0; k<indice_v; k++){	
					T par1 = T(k+1) - par1_;
					g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(0+3*i)] += exp(-powf( par1,2.)/(2*powf(par2,2.)) )*residual[l*indice_y*indice_v + j*indice_v + k]/powf(std_map_[l*indice_y+j],2);
					g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(1+3*i)] += par0*par1/powf(par2,2.) * exp(-powf(par1,2.)/(2*powf(par2,2.)) )*residual[l*indice_y*indice_v + j*indice_v + k]/powf(std_map_[l*indice_y+j],2);
					g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(2+3*i)] += par0*powf(par1, 2.)/(pow(par2,3.)) * exp(-powf(par1,2.)/(2*powf(par2,2.)) )*residual[l*indice_y*indice_v + j*indice_v + k]/powf(std_map_[l*indice_y+j],2);
				}
			}
		}
	}
	}
	}
	temps_deriv+=omp_get_wtime()-temps1_deriv;

	temps[3]+=1000*temps_conv;
	temps[2]+=1000*temps_deriv;
	temps[1]+=1000*temps_tableaux;
	temps[0]+=1000*temps_copy;
	for(int i = 0; i<4; i++){
		temps[4]+=temps[i];
	}
/*
	for(int i = 0; i<n_beta; i++){
		printf("g[%d]= %.16f\n",i, g[i]);
	}
	printf("f = %.16f\n",f);
	std::cin.ignore();
*/

	free(residual);
	free(std_map_);
	free(conv_amp);
	free(conv_mu);
	free(conv_sig);
	free(conv_conv_amp);
	free(conv_conv_mu);
	free(conv_conv_sig);
	free(image_amp);
	free(image_mu);
	free(image_sig);
}

template <typename T> 
void f_g_cube_fast_clean(parameters<T> &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, double* temps)
{
	double temps_conv;
	double temps_deriv;
	double temps_tableaux;
	double temps_copy;
	double temps_f_g_cube;

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

	double temps1_copy = omp_get_wtime();
	for(int i = 0; i< n_beta; i++){
		g[i]=0.;
	}
	f=0.;

	one_D_to_three_D_same_dimensions(beta, params, 3*M.n_gauss, indice_y, indice_x);
	//unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}
	temps_copy+= omp_get_wtime() - temps1_copy;

	double temps1_tableaux = omp_get_wtime();
	for(int j=0; j<indice_x; j++){
		for(int i=0; i<indice_y; i++){
			std::vector<T> residual_1D(indice_v,0.);
			std::vector<T> params_flat(3*M.n_gauss,0.);
			std::vector<T> cube_flat(indice_v,0.);

			for (int p=0; p<3*M.n_gauss; p++){
				params_flat[p]=params[p][i][j];
			}
			for (int p=0; p<indice_v; p++){
				cube_flat[p]=cube[j][i][p];
			}

			myresidual<T>(params_flat, cube_flat, residual_1D, M.n_gauss);

			for (int p=0;p<indice_v;p++){
				residual[j][i][p]=residual_1D[p];
			}

			if(std_map[i][j]>0.){
				f += myfunc_spec(residual_1D)*1/pow(std_map[i][j],2.); //std_map est arrondie... 
			}
		}
	}
	temps_tableaux+= omp_get_wtime() - temps1_tableaux;

	for(int i=0; i<M.n_gauss; i++){
		double temps1_conv = omp_get_wtime();
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){
				image_amp[p][q]=params[0+3*i][p][q];
				image_mu[p][q]=params[1+3*i][p][q];
				image_sig[p][q]=params[2+3*i][p][q];
			}
		}

		convolution_2D_mirror<T>(M, image_amp, conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror<T>(M, image_mu, conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror<T>(M, image_sig, conv_sig, indice_y, indice_x,3);
	
		convolution_2D_mirror<T>(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror<T>(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror<T>(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);
		temps_conv+=omp_get_wtime()-temps1_conv;
/*
		for(int p=0; p<indice_y; p++){
			for(int q=0; q<indice_x; q++){
				printf("conv_conv_amp[%d][%d] = %f\n", p,q,conv_conv_amp[p][q]);
				printf("conv_conv_mu[%d][%d] = %f\n", p,q,conv_conv_mu[p][q]);
				printf("conv_conv_sig[%d][%d] = %f\n", p,q,conv_conv_sig[p][q]);
			}
		}
*/
		double temps1_deriv = omp_get_wtime();
		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){
				f+= 0.5*M.lambda_amp*pow(conv_amp[j][l],2);
				f+= 0.5*M.lambda_mu*pow(conv_mu[j][l],2);
				f+= 0.5*M.lambda_sig*pow(conv_sig[j][l],2) + 0.5*M.lambda_var_sig*pow(image_sig[j][l]-b_params[i],2);
//				printf("b_params[i] = %f\n", b_params[i]);

				g[n_beta-M.n_gauss+i] += M.lambda_var_sig*(b_params[i]-image_sig[j][l]);
				for(int k=0; k<indice_v; k++){
					if(std_map[j][l]>0.){
						T spec = T(k+1);
						deriv[0+3*i][j][l] += exp(-pow( spec-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*residual[l][j][k]/pow(std_map[j][l],2);
						deriv[1+3*i][j][l] += params[3*i][j][l]*(spec - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( spec-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*residual[l][j][k]/pow(std_map[j][l],2);
						deriv[2+3*i][j][l] += params[3*i][j][l]*pow( spec - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( spec-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*residual[l][j][k]/pow(std_map[j][l],2);
					}
				}
				deriv[0+3*i][j][l] += M.lambda_amp*conv_conv_amp[j][l];
				deriv[1+3*i][j][l] += M.lambda_mu*conv_conv_mu[j][l];
				deriv[2+3*i][j][l] += M.lambda_sig*conv_conv_sig[j][l]+M.lambda_var_sig*(image_sig[j][l]-b_params[i]);
			}
		}
		temps_deriv+= omp_get_wtime() - temps1_deriv;
	}

	three_D_to_one_D_same_dimensions(deriv, g, 3*M.n_gauss, indice_y, indice_x);

	temps[3]+=1000*temps_conv;
	temps[2]+=1000*temps_deriv;
	temps[1]+=1000*temps_tableaux;
	temps[0]+=1000*temps_copy;
	for(int i = 0; i<4; i++){
		temps[4]+=temps[i];
	}

	}


template <typename T> 
void f_g_cube_not_very_fast_clean(parameters<T> &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, double* temps)
{
	bool print = false;

	double temps_conv;
	double temps_deriv;
	double temps_tableaux;
	double temps_copy;
	double temps_f_g_cube;

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

	double temps1_copy = omp_get_wtime();
	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}

	for(int i = 0; i< n_beta; i++){
		g[i]=0.;
	}
	f=0.;

	one_D_to_three_D_same_dimensions(beta, params, 3*M.n_gauss, indice_y, indice_x);
	//unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);
	temps_copy += omp_get_wtime()-temps1_copy;

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
			myresidual<T>(params_flat, cube_flat, residual_1D, M.n_gauss);
			for (int p=0;p<indice_v;p++){
				residual[p][i][j]=residual_1D[p];
			}
			if(std_map[i][j]>0.){
				f += myfunc_spec(residual_1D)*1/pow(std_map[i][j],2.); //std_map est arrondie... 
			}
		}
	}
	double temps2_tableaux = omp_get_wtime();

	if(print){
		printf("--> mi-chemin : f = %.16f\n", f);
	}
	double temps1_deriv = omp_get_wtime();
	int i;

	#pragma omp parallel private(i) shared(params,deriv,std_map,residual,indice_v,indice_y,indice_x)
	{
	#pragma omp for
	for(int j=0; j<indice_y; j++){
		for(int l=0; l<indice_x; l++){
			if(std_map[j][l]>0.){
				for(i=0; i<M.n_gauss; i++){
					for(int k=0; k<indice_v; k++){
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

		convolution_2D_mirror<T>(M, image_amp, conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror<T>(M, image_mu, conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror<T>(M, image_sig, conv_sig, indice_y, indice_x,3);
	
		convolution_2D_mirror<T>(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror<T>(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror<T>(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);

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

	three_D_to_one_D_same_dimensions(deriv, g, 3*M.n_gauss, indice_y, indice_x);

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;

	temps[3]+=1000*temps_conv;
	temps[2]+=1000*temps_deriv;
	temps[1]+=1000*temps_tableaux;
	temps[0]+=1000*temps_copy;
	for(int i = 0; i<4; i++){
		temps[4]+=temps[i];
	}

if(print){
	printf("--> fin-chemin : f = %.16f\n", f);
	std::cin.ignore();
	}
}


template <typename T> 
void f_g_cube_fast_clean_optim_CPU_lib(parameters<T> &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T** assist_buffer, double* temps)
{
	double temps_conv;
	double temps_deriv;
	double temps_tableaux;
	double temps_copy;
	double temps_f_g_cube;

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

	double temps1_copy = omp_get_wtime();
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
	one_D_to_three_D_same_dimensions(beta, params, 3*M.n_gauss, indice_y, indice_x);
	temps_copy += omp_get_wtime()-temps1_copy;

	//unravel_3D(beta, params, 3*M.n_gauss, indice_y, indice_x);

	double temps1_tableaux = omp_get_wtime();
	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			std::vector<T> residual_1D(indice_v,0.);
			std::vector<T> params_flat(3*M.n_gauss,0.);
			std::vector<T> cube_flat(cube[0][0].size(),0.);

			for (int p=0; p<3*M.n_gauss; p++){
				params_flat[p]=params[p][i][j];
			}
			for (int p=0; p<indice_v; p++){
				cube_flat[p]=cube[j][i][p];
			}

			myresidual<T>(params_flat, cube_flat, residual_1D, M.n_gauss);

			for (int p=0;p<indice_v;p++){
				residual[p][i][j]=residual_1D[p];
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

		convolution_2D_mirror<T>(M, image_amp, conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror<T>(M, image_mu, conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror<T>(M, image_sig, conv_sig, indice_y, indice_x,3);
	
		convolution_2D_mirror<T>(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror<T>(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror<T>(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);

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

	three_D_to_one_D_same_dimensions(deriv, g, 3*M.n_gauss, indice_y, indice_x);

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;

	temps[3]+=1000*temps_conv;
	temps[2]+=1000*temps_deriv;
	temps[1]+=1000*temps_tableaux;
	temps[0]+=1000*temps_copy;
	for(int i = 0; i<4; i++){
		temps[4]+=temps[i];
	}
}



template <typename T> 
void f_g_cube_cuda_L_clean(parameters<T> &M, T& f, T* g, int n, std::vector<std::vector<std::vector<T>>> &cube, T* beta, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened, double* temps, double temps_transfert_d, double temps_mirroirs)
{
	double temps_conv;
	double temps_deriv;
	double temps_tableaux;
	double temps_copy;
	double temps_f_g_cube;

//	init_templates();
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

	double temps1_copy = omp_get_wtime();
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
	temps_copy += omp_get_wtime()-temps1_copy;

//exit(0);

/*
	std::cout<< "	-> Temps d'exécution transfert données : " << temps_copy  <<std::endl;
	std::cout<< "	-> Temps d'exécution attache aux données : " << temps_tableaux <<std::endl;
	std::cout<< "	-> Temps d'exécution deriv : " << temps_deriv  <<std::endl;
	std::cout<< "	-> Temps d'exécution régularisation : " << temps_conv <<std::endl;
*/

	double temps1_tableaux = omp_get_wtime();
	f =  compute_residual_and_f<T>(beta, taille_beta_modif, product_beta, cube_flattened, taille_cube, product_cube, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, indice_x, indice_y, indice_v, M.n_gauss);
	double temps2_tableaux = omp_get_wtime();

	double temps1_deriv = omp_get_wtime();
	gradient_L_2_beta<T>(deriv, taille_deriv, product_deriv, beta, taille_beta_modif, product_beta_modif, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, M.n_gauss);
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
		if(false){//indice_x>=128 || indice_y>=128){//true){//
			conv2D_GPU(image_amp, Kernel, conv_amp, indice_x, indice_y, temps_transfert_d, temps_mirroirs);
			conv2D_GPU(image_mu, Kernel, conv_mu, indice_x, indice_y, temps_transfert_d, temps_mirroirs);
			conv2D_GPU(image_sig, Kernel, conv_sig, indice_x, indice_y, temps_transfert_d, temps_mirroirs);

			conv2D_GPU(conv_amp, Kernel, conv_conv_amp, indice_x, indice_y, temps_transfert_d, temps_mirroirs);
			conv2D_GPU(conv_mu, Kernel, conv_conv_mu, indice_x, indice_y, temps_transfert_d, temps_mirroirs);
			conv2D_GPU(conv_sig, Kernel, conv_conv_sig, indice_x, indice_y, temps_transfert_d, temps_mirroirs);
		} else{
			convolution_2D_mirror_flat<T>(M, image_amp, conv_amp, indice_y, indice_x,3);
			convolution_2D_mirror_flat<T>(M, image_mu, conv_mu, indice_y, indice_x,3);
			convolution_2D_mirror_flat<T>(M, image_sig, conv_sig, indice_y, indice_x,3);

			convolution_2D_mirror_flat<T>(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
			convolution_2D_mirror_flat<T>(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
			convolution_2D_mirror_flat<T>(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);
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

	for(int i=0; i<n_beta-M.n_gauss; i++){
		g[i]=deriv[i];
	}

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

	temps[3]+=1000*temps_conv;
	temps[2]+=1000*temps_deriv;
	temps[1]+=1000*temps_tableaux;
	temps[0]+=1000*temps_copy;
	for(int i = 0; i<4; i++){
		temps[4]+=temps[i];
	}

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



template <typename T> 
void f_g_cube_cuda_L_clean_lib(parameters<T> &M, T &f, T* g, int n, T* beta, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened, double* temps, double temps_transfert_d, double temps_mirroirs)
{
	double temps_conv;
	double temps_deriv;
	double temps_tableaux;
	double temps_copy;
	double temps_f_g_cube;

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

	double temps1_copy = omp_get_wtime();
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
	temps_copy += omp_get_wtime()-temps1_copy;

	double temps1_tableaux = omp_get_wtime();
	f =  compute_residual_and_f<T>(beta, taille_beta_modif, product_beta, cube_flattened, taille_cube, product_cube, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, indice_x, indice_y, indice_v, M.n_gauss);
	double temps2_tableaux = omp_get_wtime();

	double temps1_deriv = omp_get_wtime();
	gradient_L_2_beta<T>(deriv, taille_deriv, product_deriv, beta, taille_beta_modif, product_beta_modif, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, M.n_gauss);
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
			conv2D_GPU(image_amp, Kernel, conv_amp, indice_x, indice_y, temps_transfert_d, temps_mirroirs);
			conv2D_GPU(image_mu, Kernel, conv_mu, indice_x, indice_y, temps_transfert_d, temps_mirroirs);
			conv2D_GPU(image_sig, Kernel, conv_sig, indice_x, indice_y, temps_transfert_d, temps_mirroirs);

			conv2D_GPU(conv_amp, Kernel, conv_conv_amp, indice_x, indice_y, temps_transfert_d, temps_mirroirs);
			conv2D_GPU(conv_mu, Kernel, conv_conv_mu, indice_x, indice_y, temps_transfert_d, temps_mirroirs);
			conv2D_GPU(conv_sig, Kernel, conv_conv_sig, indice_x, indice_y, temps_transfert_d, temps_mirroirs);
		} else{
			convolution_2D_mirror_flat<T>(M, image_amp, conv_amp, indice_y, indice_x,3);
			convolution_2D_mirror_flat<T>(M, image_mu, conv_mu, indice_y, indice_x,3);
			convolution_2D_mirror_flat<T>(M, image_sig, conv_sig, indice_y, indice_x,3);

			convolution_2D_mirror_flat<T>(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
			convolution_2D_mirror_flat<T>(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
			convolution_2D_mirror_flat<T>(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);
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
	for(int i=0; i<n_beta-M.n_gauss; i++){
		g[i]=deriv[i];
	}

	temps_conv+= temps2_conv - temps1_conv;
	temps_deriv+= temps2_deriv - temps1_deriv;
	temps_tableaux += temps2_tableaux - temps1_tableaux;

	temps[3]+=1000*temps_conv;
	temps[2]+=1000*temps_deriv;
	temps[1]+=1000*temps_tableaux;
	temps[0]+=1000*temps_copy;
	for(int i = 0; i<4; i++){
		temps[4]+=temps[i];
	}
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

template <typename T> 
void convolution_2D_mirror_flat(const parameters<T> &M, T* image, T* conv, int dim_y, int dim_x, int dim_k)
{
	T kernel[]= {0.,-0.25,0.,-0.25,1.,-0.25,0.,-0.25,0.}; 
	int ii(0),jj(0),mm(0),nn(0),kCenterY(0), kCenterX(0);

	std::vector<std::vector<T>> ext_conv(dim_y+4, std::vector<T>(dim_x+4,0.));
	std::vector<std::vector<T>> extended(dim_y+4, std::vector<T>(dim_x+4,0.));


	for(int j(0); j<dim_y; j++)
	{
		for(int i(0); i<dim_x; i++)
		{
			extended[2+j][2+i]=image[dim_x*j+i];
		}
	}


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
						ext_conv[i-1][j-1] += extended[ii-1][jj-1]*kernel[(mm-1)*3+nn-1];
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

template <typename T> 
void convolution_2D_mirror(const parameters<T> &M, const std::vector<std::vector<T>> &image, std::vector<std::vector<T>> &conv, int dim_y, int dim_x, int dim_k)
{
	T kernel[]= {0.,-0.25,0.,-0.25,1.,-0.25,0.,-0.25,0.}; 
	int ii(0),jj(0),mm(0),nn(0),kCenterY(0), kCenterX(0);

	std::vector <std::vector<T>> ext_conv(dim_y+4, std::vector<T>(dim_x+4,0.));
	std::vector <std::vector<T>> extended(dim_y+4, std::vector<T>(dim_x+4,0.));

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
						ext_conv[i-1][j-1] += extended[ii-1][jj-1]*kernel[(mm-1)*3+nn-1];
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

template <typename T> 
void one_D_to_three_D_same_dimensions(T* vector, std::vector<std::vector<std::vector<T>>> &cube_3D, int dim_x, int dim_y, int dim_v)
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


template <typename T> 
void three_D_to_one_D_same_dimensions(const std::vector<std::vector<std::vector<T>>> &cube_3D, T* vector, int dim_x, int dim_y, int dim_v)
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

template <typename T> 
T myfunc_spec(std::vector<T> &residual) {
	T S(0.);
	for(int p(0); p<residual.size(); p++) {
		S+=pow(residual[p],2);
	}
	return 0.5*S;
}

template <typename T> 
T model_function(int x, T a, T m, T s) {
	return a*exp(-pow(T(x)-m,2.) / (2.*pow(s,2.)));
}

template <typename T> 
void myresidual(std::vector<T> &params, std::vector<T> &line, std::vector<T> &residual, int n_gauss_i) {
	int i,k;
	std::vector<T> model(residual.size(),0.);
	for(i=0; i<n_gauss_i; i++) {
		for(k=0; k<residual.size(); k++) {
			int nu = k+1;
			model[k]+= model_function<T>(nu, params[0+3*i], params[1+3*i], params[2+3*i]);
		}
	}
	for(int p=0; p<residual.size(); p++) {
		residual[p]=model[p]-line[p]; 
	}
}



template double model_function(int x, double a, double m, double s);
template void f_g_cube_fast_unidimensional(parameters<double>&, double&, double*, int, double*, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, double*, double*);	

template void convolution_2D_mirror_flat(const parameters<double> &M, double* image, double* conv, int dim_y, int dim_x, int dim_k);
template void convolution_2D_mirror(const parameters<double> &M, const std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &conv, int dim_y, int dim_x, int dim_k);

template void f_g_cube_fast_clean(parameters<double>&, double &, double*, int n, std::vector<std::vector<std::vector<double>>>&, double*, int , int , int , std::vector<std::vector<double>>&, double*);
template void f_g_cube_not_very_fast_clean(parameters<double>&, double &, double*, int n, std::vector<std::vector<std::vector<double>>>&, double*, int , int , int , std::vector<std::vector<double>>&, double*);
template void f_g_cube_fast_clean_optim_CPU_lib(parameters<double>&, double &, double*, int n, std::vector<std::vector<std::vector<double>>>&, double*, int , int , int , std::vector<std::vector<double>>&, double**, double*);
template void f_g_cube_cuda_L_clean(parameters<double>&, double &, double*, int n, std::vector<std::vector<std::vector<double>>>&, double*, int , int , int , std::vector<std::vector<double>>&, double*, double*, double, double);
template void f_g_cube_cuda_L_clean_lib(parameters<double>&, double &, double*, int, double*, int , int , int , std::vector<std::vector<double>>&, double*, double*, double, double);

template void one_D_to_three_D_same_dimensions(double*, std::vector<std::vector<std::vector<double>>>&, int, int, int);
template void three_D_to_one_D_same_dimensions(const std::vector<std::vector<std::vector<double>>>&, double*, int, int, int);

template double myfunc_spec(std::vector<double>&);
template void myresidual(std::vector<double>&, std::vector<double>&, std::vector<double>&, int);






template float model_function(int x, float a, float m, float s);
template void f_g_cube_fast_unidimensional(parameters<float>&, float&, float*, int, float*, std::vector<std::vector<std::vector<float>>>&, float*, int, int, int, float*, double*);	

template void convolution_2D_mirror_flat(const parameters<float> &M, float* image, float* conv, int dim_y, int dim_x, int dim_k);
template void convolution_2D_mirror(const parameters<float> &M, const std::vector<std::vector<float>> &image, std::vector<std::vector<float>> &conv, int dim_y, int dim_x, int dim_k);

template void f_g_cube_fast_clean(parameters<float>&, float &, float*, int n, std::vector<std::vector<std::vector<float>>>&, float*, int , int , int , std::vector<std::vector<float>>&, double*);
template void f_g_cube_not_very_fast_clean(parameters<float>&, float &, float*, int n, std::vector<std::vector<std::vector<float>>>&, float*, int , int , int , std::vector<std::vector<float>>&, double*);
template void f_g_cube_fast_clean_optim_CPU_lib(parameters<float>&, float &, float*, int n, std::vector<std::vector<std::vector<float>>>&, float*, int , int , int , std::vector<std::vector<float>>&, float**, double*);
template void f_g_cube_cuda_L_clean(parameters<float>&, float &, float*, int n, std::vector<std::vector<std::vector<float>>>&, float*, int , int , int , std::vector<std::vector<float>>&, float*, double*, double, double);
template void f_g_cube_cuda_L_clean_lib(parameters<float>&, float &, float*, int, float*, int , int , int , std::vector<std::vector<float>>&, float*, double*, double, double);

template void one_D_to_three_D_same_dimensions(float*, std::vector<std::vector<std::vector<float>>>&, int, int, int);
template void three_D_to_one_D_same_dimensions(const std::vector<std::vector<std::vector<float>>>&, float*, int, int, int);

template float myfunc_spec(std::vector<float>&);
template void myresidual(std::vector<float>&, std::vector<float>&, std::vector<float>&, int);



