#include "f_g_cube.hpp"
//#include "f_g_cube_gpu.hpp"
#include "kernels_for_hybrid.cu"

#define print false

template <typename T> 
T f_g_cube_fast_unidimensional_return(parameters<T> &M, T* g, int n, T* cube, std::vector<std::vector<std::vector<T>>>& cube_for_cache, T* beta, int indice_v, int indice_y, int indice_x, T* std_map, double* temps){

	T f = 0.;

/*
	for(int i = 0; i<6; i++){
		printf("beta[%d]= %.16f\n",i, beta[i]);
	}
*/
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

	printf("--->f= %.16f\n", f);


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

	printf("--->f= %.16f\n", f);

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
					g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(0+3*i)] += exp(-powf( par1,2.)/(2*powf(par2,2.)) )*(residual[l*indice_y*indice_v + j*indice_v + k]/powf(std_map_[l*indice_y+j],2));
					g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(1+3*i)] += par0*par1/powf(par2,2.) * exp(-powf(par1,2.)/(2*powf(par2,2.)) )*(residual[l*indice_y*indice_v + j*indice_v + k]/powf(std_map_[l*indice_y+j],2));
					g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(2+3*i)] += par0*powf(par1, 2.)/(pow(par2,3.)) * exp(-powf(par1,2.)/(2*powf(par2,2.)) )*(residual[l*indice_y*indice_v + j*indice_v + k]/powf(std_map_[l*indice_y+j],2));
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

	return f;
}


















template <typename T> 
void f_g_cube_fast_unidimensional(parameters<T> &M, T &f, T* __restrict__ g, int n, T* __restrict__ cube, std::vector<std::vector<std::vector<T>>>& cube_for_cache, T* __restrict__ beta, int indice_v, int indice_y, int indice_x, T* __restrict__ std_map, double* temps){	
//void f_g_cube_fast_unidimensional(parameters<T> &M, T &f, T* g, int n, T* cube, std::vector<std::vector<std::vector<T>>>& cube_for_cache, T* beta, int indice_v, int indice_y, int indice_x, T* std_map, double* temps){


/*
	printf("beta[0]= %.16f\n", beta[0]);
	for(int i = 0; i<4; i++){
		printf("beta[%d]= %.16f\n",i, beta[i]);
	}
*/
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

//	omp_set_num_threads(50);
	omp_set_num_threads(1);
	#pragma omp parallel private(j,i) shared(hypercube_tilde, residual ,par,M,beta,std_map_,indice_v,indice_y,indice_x)
	{
//	printf("num threads = %d\n", omp_get_num_threads());
	#pragma omp for
	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
			for (int p=0; p<indice_v; p++){
				T spec = T(p+1); //+1
				T accu = 0.;
				for (int p_par=0; p_par<M.n_gauss; p_par++){
					par[0]= beta[j*3*M.n_gauss*indice_y+i*3*M.n_gauss+(3*p_par+0)];
					par[1]= beta[j*3*M.n_gauss*indice_y+i*3*M.n_gauss+(3*p_par+1)];
					par[2]= beta[j*3*M.n_gauss*indice_y+i*3*M.n_gauss+(3*p_par+2)];
					accu+= par[0]*exp(-pow(spec-par[1],2)/(2*pow(par[2],2.)));
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
					accu+=pow(residual[j*indice_y*indice_v+i*indice_v+p],2.);
				}
	//				printf("accu = %f\n",accu);
				f += 0.5*accu/pow(std_map_[j*indice_y+i],2.); //std_map est arrondie... 
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
				f+= 0.5*M.lambda_amp*pow(conv_amp[j*indice_x+l],2);
				f+= 0.5*M.lambda_mu*pow(conv_mu[j*indice_x+l],2);
				f+= 0.5*M.lambda_sig*pow(conv_sig[j*indice_x+l],2) + 0.5*M.lambda_var_sig*pow(image_sig[j*indice_x+l]-b_params[i],2);
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
						T par1 = T(k+1) - par1_;//+1   T(p+1)
						g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(0+3*i)] += exp(-pow( par1,2.)/(2*pow(par2,2.)) )*residual[l*indice_y*indice_v + j*indice_v + k]/pow(std_map_[l*indice_y+j],2);
						g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(1+3*i)] += par0*par1/pow(par2,2.) * exp(-pow(par1,2.)/(2*pow(par2,2.)) )*residual[l*indice_y*indice_v + j*indice_v + k]/pow(std_map_[l*indice_y+j],2);
						g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(2+3*i)] += par0*pow(par1, 2.)/(pow(par2,3.)) * exp(-pow(par1,2.)/(2*pow(par2,2.)) )*residual[l*indice_y*indice_v + j*indice_v + k]/pow(std_map_[l*indice_y+j],2);
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
	printf("-->f= %.16f\n", f);
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
void f_g_cube_fast_unidimensional_test(parameters<T> &M, T &f, T* g, int n, T* cube, std::vector<std::vector<std::vector<T>>>& cube_for_cache, T* beta, int indice_v, int indice_y, int indice_x, T* std_map, double* temps){

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
				T spec = T(p+1); //+1
				T accu = 0.;
				for (int p_par=0; p_par<M.n_gauss; p_par++){
					par[0]= beta[j*3*M.n_gauss*indice_y+i*3*M.n_gauss+(3*p_par+0)];
					par[1]= beta[j*3*M.n_gauss*indice_y+i*3*M.n_gauss+(3*p_par+1)];
					par[2]= beta[j*3*M.n_gauss*indice_y+i*3*M.n_gauss+(3*p_par+2)];
					accu+= par[0]*exp(-pow(spec-par[1],2)/(2*pow(par[2],2.)));
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
					accu+=pow(residual[j*indice_y*indice_v+i*indice_v+p],2.);
				}
	//				printf("accu = %f\n",accu);
				f += 0.5*accu/pow(std_map_[j*indice_y+i],2.); //std_map est arrondie... 
			}
		}
	}



	temps_tableaux+=omp_get_wtime()-temps1_tableaux;
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
						T par1 = T(k+1) - par1_;//+1   T(p+1)
						g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(0+3*i)] += exp(-pow( par1,2.)/(2*pow(par2,2.)) )*residual[l*indice_y*indice_v + j*indice_v + k]/pow(std_map_[l*indice_y+j],2);
						g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(1+3*i)] += par0*par1/pow(par2,2.) * exp(-pow(par1,2.)/(2*pow(par2,2.)) )*residual[l*indice_y*indice_v + j*indice_v + k]/pow(std_map_[l*indice_y+j],2);
						g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(2+3*i)] += par0*pow(par1, 2.)/(pow(par2,3.)) * exp(-pow(par1,2.)/(2*pow(par2,2.)) )*residual[l*indice_y*indice_v + j*indice_v + k]/pow(std_map_[l*indice_y+j],2);
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
	printf("-->f= %.16f\n", f);
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

	one_D_to_three_D_inverted_dimensions(beta, params, 3*M.n_gauss, indice_y, indice_x);
	
/*
	for(int i = 0; i< 13; i++){
		printf("beta[%d] = %.16f\n",i,beta[i]);
	}
	printf("params[0][0][0] = %.16f\n",params[0][0][0]);
	printf("params[0][1][0] = %.16f\n",params[0][1][0]);
	printf("params[0][0][1] = %.16f\n",params[0][0][1]);
	printf("params[1][0][0] = %.16f\n",params[1][0][0]);
	printf("params[1][1][0] = %.16f\n",params[1][1][0]);
	printf("params[1][0][1] = %.16f\n",params[1][0][1]);
*/

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


//	printf("f = %.26f\n", f);


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

		if(false){
			for(int p=0; p<indice_y; p++){
				for(int q=0; q<indice_x; q++){
					printf("std_map[%d][%d] = %.26f\n", p,q,std_map[p][q]);
				}
			}
			for(int p=0; p<indice_y; p++){
				for(int q=0; q<indice_x; q++){
					printf("conv_conv_amp[%d][%d] = %.26f\n", p,q,conv_conv_amp[p][q]);
					printf("conv_conv_mu[%d][%d] = %.26f\n", p,q,conv_conv_mu[p][q]);
					printf("conv_conv_sig[%d][%d] = %.26f\n", p,q,conv_conv_sig[p][q]);
					printf("conv_amp[%d][%d] = %.26f\n", p,q,conv_amp[p][q]);
					printf("conv_mu[%d][%d] = %.26f\n", p,q,conv_mu[p][q]);
					printf("conv_sig[%d][%d] = %.26f\n", p,q,conv_sig[p][q]);
					printf("image_amp[%d][%d] = %.26f\n", p,q,image_amp[p][q]);
					printf("image_mu[%d][%d] = %.26f\n", p,q,image_mu[p][q]);
					printf("image_sig[%d][%d] = %.26f\n", p,q,image_sig[p][q]);
				}
			}
		}

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
						deriv[0+3*i][j][l] += exp(-pow( spec-params[1+3*i][j][l],2)/(2*pow(params[2+3*i][j][l],2)) )*(residual[l][j][k]/pow(std_map[j][l],2));
						deriv[1+3*i][j][l] += params[3*i][j][l]*(spec - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2) * exp(-pow( spec-params[1+3*i][j][l],2)/(2*pow(params[2+3*i][j][l],2)) )*(residual[l][j][k]/pow(std_map[j][l],2));
						deriv[2+3*i][j][l] += params[3*i][j][l]*(pow( spec - params[1+3*i][j][l], 2)/(pow(params[2+3*i][j][l],3))) * exp(-pow( spec-params[1+3*i][j][l],2)/(2*pow(params[2+3*i][j][l],2)) )*(residual[l][j][k]/pow(std_map[j][l],2));
					}
				}
			}
		}

/*
		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){
				printf("deriv[%d][%d][%d] = %.26f\n",0+3*i, j, l,deriv[0+3*i][j][l]);
				printf("deriv[%d][%d][%d] = %.26f\n",1+3*i, j, l,deriv[1+3*i][j][l]);
				printf("deriv[%d][%d][%d] = %.26f\n",2+3*i, j, l,deriv[2+3*i][j][l]);
			}
		}		
		int j = 0;
		int l = 0;
		int k = 33;
			printf("index = %d\n", k);
			printf("scalar test = %.26f\n",params[3*i][j][l]*(T(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2) * exp(-pow( T(k+1)-params[1+3*i][j][l],2)/	(2*pow(params[2+3*i][j][l],2)) )*(residual[l][j][k]/pow(std_map[j][l],2)));
			printf("scalar test = %.26f\n",params[3*i][j][l]*(T(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2) * exp(-pow( T(k+1)-params[1+3*i][j][l],2)/	(2*pow(params[2+3*i][j][l],2)) ));
			printf("scalar test = %.26f\n",params[3*i][j][l]*(T(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2));
			printf("scalar test = %.26f\n",residual[l][j][k]);
			printf("scalar test = %.26f\n",pow(std_map[j][l],2));
			printf("scalar test = %.26f\n",1/pow(std_map[j][l],2));
			printf("scalar test = %.26f\n",residual[l][j][k]/pow(std_map[j][l],2));

			printf("------------------------------------\n");

			printf("scalar test = %.26f\n",params[3*i][j][l]*(T(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2) * exp(-pow( T(k+1)-params[1+3*i][j][l],2)/(2*pow(params[2+3*i][j][l],2)) )*(residual[l][j][k]/pow(std_map[j][l],2)));
			printf("scalar test = %.26f\n",params[3*i][j][l]*(T(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2) * exp(-pow( T(k+1)-params[1+3*i][j][l],2)/(2*pow(params[2+3*i][j][l],2)) ));
			printf("scalar test = %.26f\n",params[3*i][j][l]*(T(k+1) - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2));
			printf("------------------------------------\n");

			printf("scalar test = %.26f\n",exp(-pow( T(k+1)-params[1+3*i][j][l],2)/(2*pow(params[2+3*i][j][l],2)) )*(residual[l][j][k]/pow(std_map[j][l],2)));
			printf("scalar test = %.26f\n",exp(-pow( T(k+1)-params[1+3*i][j][l],2)/(2*pow(params[2+3*i][j][l],2)) ));
			exit(0);
*/


		for(int j=0; j<indice_y; j++){
			for(int l=0; l<indice_x; l++){
				deriv[0+3*i][j][l] += M.lambda_amp*conv_conv_amp[j][l];
				deriv[1+3*i][j][l] += M.lambda_mu*conv_conv_mu[j][l];
				deriv[2+3*i][j][l] += M.lambda_sig*conv_conv_sig[j][l]+M.lambda_var_sig*(image_sig[j][l]-b_params[i]);

			}
		}
		temps_deriv+= omp_get_wtime() - temps1_deriv;
	}

	three_D_to_one_D_inverted_dimensions(deriv, g, 3*M.n_gauss, indice_y, indice_x);
//	three_D_to_one_D_same_dimensions(deriv, g, 3*M.n_gauss, indice_y, indice_x);

//	printf("f = %.16f\n", f);

	temps[3]+=1000*temps_conv;
	temps[2]+=1000*temps_deriv;
	temps[1]+=1000*temps_tableaux;
	temps[0]+=1000*temps_copy;
	for(int i = 0; i<4; i++){
		temps[4]+=temps[i];
	}

//	exit(0);

}


template <typename T> 
void f_g_cube_not_very_fast_clean(parameters<T> &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, double* temps)
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
/*
	for(int i = 0; i<n_beta; i++){
		printf("beta[%d] = %f\n",i,beta[i]);
	}
*/
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
void f_g_cube_cuda_L_clean_templatized(parameters<T> &M, T& f, T* g, int n, T* beta, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened, double* temps, double temps_transfert_d, double temps_mirroirs, float* temps_kernel)
{
//	dummyInstantiator_sort();
	double temps_conv;
	double temps_deriv;
	double temps_tableaux;
	double temps_copy;
	double temps_f_g_cube;

//	bool print = false;	
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
    
//	std::vector<T> b_params(M.n_gauss,0.);
	int i,k,j,l,p;
	double temps1_copy = omp_get_wtime();

	int taille_params_flat[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_deriv[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_residual[] = {indice_v, indice_y, indice_x};
	int taille_std_map_[] = {indice_y, indice_x};
	int taille_beta[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_beta_modif[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_cube[] = {indice_v, indice_y, indice_x};

	int product_residual = taille_residual[0]*taille_residual[1]*taille_residual[2];
	int product_params_flat = taille_params_flat[0]*taille_params_flat[1]*taille_params_flat[2];
	int product_deriv = taille_deriv[0]*taille_deriv[1]*taille_deriv[2];
	int product_std_map_ = taille_std_map_[0]*taille_std_map_[1];
	int product_cube = taille_cube[0]*taille_cube[1]*taille_cube[2]; 
	int product_beta = taille_beta[0]*taille_beta[1]*taille_beta[2];
	int product_beta_modif = taille_beta[0]*taille_beta[1]*taille_beta[2];

	size_t size_deriv = product_deriv * sizeof(T);
	size_t size_res = product_residual * sizeof(T);
	size_t size_std = product_std_map_ * sizeof(T);
	size_t size_beta_modif = product_beta_modif * sizeof(T);

	T* deriv = (T*)malloc(size_deriv);
	T* residual = (T*)malloc(size_res);
	T* std_map_ = (T*)malloc(size_std);

	T* b_params = (T*)malloc(M.n_gauss*sizeof(T));

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;

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


    T* beta_dev = NULL;
    checkCudaErrors(cudaMalloc(&beta_dev, n*sizeof(T)));
    checkCudaErrors(cudaMemcpy(beta_dev, beta, n*sizeof(T), cudaMemcpyHostToDevice));

    T* residual_dev = NULL;
    checkCudaErrors(cudaMalloc(&residual_dev, indice_x*indice_y*indice_v*sizeof(T)));
    checkCudaErrors(cudaMemcpy(residual_dev, cube_flattened, indice_x*indice_y*indice_v*sizeof(T), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
//	checkCudaErrors(cudaMemcpy(residual_dev, cube_flattened, indice_x*indice_y*indice_v*sizeof(T), cudaMemcpyHostToDevice));

    T* std_map_dev = NULL;
    checkCudaErrors(cudaMalloc(&std_map_dev, indice_x*indice_y*sizeof(T)));
    checkCudaErrors(cudaMemcpy(std_map_dev, std_map_, indice_x*indice_y*sizeof(T), cudaMemcpyHostToDevice));

    T* g_dev = NULL;
    checkCudaErrors(cudaMalloc(&g_dev, n*sizeof(T)));
    checkCudaErrors(cudaMemset(g_dev, 0., n*sizeof(T)));

	temps_copy += omp_get_wtime()-temps1_copy;
    checkCudaErrors(cudaDeviceSynchronize());

	double temps1_tableaux = omp_get_wtime();

	float tmp_temps_kernel_res[3] = {0.,0.,0.};
	f =  compute_residual_and_f_less_memory<T>(beta_dev, taille_beta_modif, product_beta, cube_flattened, taille_cube, product_cube, residual_dev, taille_residual, product_residual, std_map_dev, taille_std_map_, product_std_map_, indice_x, indice_y, indice_v, M.n_gauss,tmp_temps_kernel_res);
	double temps2_tableaux = omp_get_wtime();
	temps_kernel[0] += tmp_temps_kernel_res[0]/1000;//compute_residual
	temps_kernel[1] += tmp_temps_kernel_res[1]/1000;//compute_Q_map
	temps_kernel[2] += tmp_temps_kernel_res[2]/1000;//reduction_loop

//	printf("cpu : %.16f\n",temps2_tableaux-temps1_tableaux);

	double temps1_deriv = omp_get_wtime();

	float tmp_temps_kernel_grad[1]={0.};
	gradient_L_2_beta<T>(g_dev, taille_deriv, product_deriv, beta_dev, taille_beta_modif, product_beta_modif, residual_dev, taille_residual, product_residual, std_map_dev, taille_std_map_, product_std_map_, M.n_gauss, tmp_temps_kernel_grad);
	temps_kernel[3] += tmp_temps_kernel_grad[0]/1000;//compute_nabla_Q

/*
	printf("tmp_temps_kernel_res[0] = %.16f\n",tmp_temps_kernel_res[0]);
	printf("tmp_temps_kernel_res[1] = %.16f\n",tmp_temps_kernel_res[1]);
	printf("tmp_temps_kernel_res[2] = %.16f\n",tmp_temps_kernel_res[2]);
	printf("tmp_temps_kernel_grad[0] = %.16f\n",tmp_temps_kernel_grad[0]);
*/
	double temps2_deriv = omp_get_wtime();

	if(print){
		printf("Milieu :\n");
		printf("-> f = %.16f\n", f);
		std::cin.ignore();
	}


	double temps1_conv = omp_get_wtime();

	float tmp_temps_kernel_regu[8] = {0.,0.,0.,0.,0.,0.,0.,0.};
	regularization<T>(beta_dev, g_dev, b_params, f, indice_x, indice_y, indice_v, M, tmp_temps_kernel_regu);
	temps_kernel[4] += tmp_temps_kernel_regu[0]/1000;//get_gaussian_parameter_maps
	temps_kernel[5] += tmp_temps_kernel_regu[1]/1000;//perform_mirror_effect_before_convolution
	temps_kernel[6] += tmp_temps_kernel_regu[2]/1000;//ConvKernel
	temps_kernel[7] += tmp_temps_kernel_regu[3]/1000;//copy_gpu
	temps_kernel[8] += tmp_temps_kernel_regu[4]/1000;//compute_R_map
	temps_kernel[9] += tmp_temps_kernel_regu[5]/1000;//compute_nabla_R_wrt_theta
	temps_kernel[10] += tmp_temps_kernel_regu[6]/1000;//compute_nabla_R_wrt_m
	temps_kernel[2] += tmp_temps_kernel_regu[7]/1000;//reduction

	double temps2_conv = omp_get_wtime();

    checkCudaErrors(cudaMemcpy(g, g_dev, n*sizeof(T), cudaMemcpyDeviceToHost));

	if (false)
	{
		lim = n_beta;
		for(int i=0; i<lim; i++){
			printf("g[%d] = %.16f\n",i, g[i]);
		}/*		for(int i=lim-40; i<lim; i++){
			printf("g[%d] = %.16f\n",i, g[i]);
		}*/

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

	free(residual);
	free(std_map_);
	free(b_params);

	checkCudaErrors(cudaFree(beta_dev));
	checkCudaErrors(cudaFree(residual_dev));
	checkCudaErrors(cudaFree(std_map_dev));
    checkCudaErrors(cudaFree(g_dev));
}




template <typename T> 
void f_g_cube_cuda_L_clean_templatized_less_transfers(parameters<T> &M, T& f, T* g, int n, T* beta, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened_dev, double* temps, double temps_transfert_d, double temps_mirroirs, float* temps_kernel)
{

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  cudaEventRecord(start);

//	dummyInstantiator_sort();
	double temps_conv;
	double temps_deriv;
	double temps_tableaux;
	double temps_copy;
	double temps_f_g_cube;

	float tmp_transfer = 0.;

//	bool print = false;	
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
    
//	std::vector<T> b_params(M.n_gauss,0.);
	int i,k,j,l,p;

	int taille_params_flat[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_deriv[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_residual[] = {indice_v, indice_y, indice_x};
	int taille_std_map_[] = {indice_y, indice_x};
	int taille_beta[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_beta_modif[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_cube[] = {indice_v, indice_y, indice_x};

	int product_residual = taille_residual[0]*taille_residual[1]*taille_residual[2];
	int product_params_flat = taille_params_flat[0]*taille_params_flat[1]*taille_params_flat[2];
	int product_deriv = taille_deriv[0]*taille_deriv[1]*taille_deriv[2];
	int product_std_map_ = taille_std_map_[0]*taille_std_map_[1];
	int product_cube = taille_cube[0]*taille_cube[1]*taille_cube[2]; 
	int product_beta = taille_beta[0]*taille_beta[1]*taille_beta[2];
	int product_beta_modif = taille_beta[0]*taille_beta[1]*taille_beta[2];

	size_t size_deriv = product_deriv * sizeof(T);
	size_t size_res = product_residual * sizeof(T);
	size_t size_std = product_std_map_ * sizeof(T);
	size_t size_beta_modif = product_beta_modif * sizeof(T);

	T* residual = (T*)malloc(size_res);
	T* std_map_ = (T*)malloc(size_std);

	T* b_params = (T*)malloc(M.n_gauss*sizeof(T));

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;

	double temps1_copy = omp_get_wtime();

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


    T* beta_dev = NULL;
    checkCudaErrors(cudaMalloc(&beta_dev, n*sizeof(T)));
    checkCudaErrors(cudaMemcpy(beta_dev, beta, n*sizeof(T), cudaMemcpyHostToDevice));

    T* residual_dev = NULL;
    checkCudaErrors(cudaMalloc(&residual_dev, indice_x*indice_y*indice_v*sizeof(T)));
    checkCudaErrors(cudaDeviceSynchronize());
    copy_dev<T><<<ceil(float(indice_x*indice_y*indice_v)/float(BLOCK_SIZE_REDUCTION)),BLOCK_SIZE_REDUCTION>>>(cube_flattened_dev, residual_dev, indice_x*indice_y*indice_v);
    checkCudaErrors(cudaDeviceSynchronize());
//	checkCudaErrors(cudaMemcpy(residual_dev, cube_flattened, indice_x*indice_y*indice_v*sizeof(T), cudaMemcpyHostToDevice));

    T* std_map_dev = NULL;
    checkCudaErrors(cudaMalloc(&std_map_dev, indice_x*indice_y*sizeof(T)));
    checkCudaErrors(cudaMemcpy(std_map_dev, std_map_, indice_x*indice_y*sizeof(T), cudaMemcpyHostToDevice));

    T* g_dev = NULL;
    checkCudaErrors(cudaMalloc(&g_dev, n*sizeof(T)));
    checkCudaErrors(cudaMemset(g_dev, 0., n*sizeof(T)));

	temps_copy += omp_get_wtime()-temps1_copy;

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventElapsedTime(&tmp_transfer, start, stop));
    checkCudaErrors(cudaDeviceSynchronize());

	


	double temps1_tableaux = omp_get_wtime();

	float tmp_temps_kernel_res[3] = {0.,0.,0.};
	f =  compute_residual_and_f_less_memory<T>(beta_dev, taille_beta_modif, product_beta, cube_flattened_dev, taille_cube, product_cube, residual_dev, taille_residual, product_residual, std_map_dev, taille_std_map_, product_std_map_, indice_x, indice_y, indice_v, M.n_gauss,tmp_temps_kernel_res);
//	f =  compute_residual_and_f<T>(beta, taille_beta_modif, product_beta, cube_flattened, taille_cube, product_cube, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, indice_x, indice_y, indice_v, M.n_gauss);
	double temps2_tableaux = omp_get_wtime();


	temps_kernel[0] += tmp_temps_kernel_res[0]/1000;//compute_residual
	temps_kernel[1] += tmp_temps_kernel_res[1]/1000;//compute_Q_map
	temps_kernel[2] += tmp_temps_kernel_res[2]/1000;//reduction_loop

//	printf("cpu : %.16f\n",temps2_tableaux-temps1_tableaux);

	double temps1_deriv = omp_get_wtime();

	float tmp_temps_kernel_grad[1]={0.};
	gradient_L_2_beta<T>(g_dev, taille_deriv, product_deriv, beta_dev, taille_beta_modif, product_beta_modif, residual_dev, taille_residual, product_residual, std_map_dev, taille_std_map_, product_std_map_, M.n_gauss, tmp_temps_kernel_grad);
	temps_kernel[3] += tmp_temps_kernel_grad[0]/1000;//compute_nabla_Q

/*
	printf("tmp_temps_kernel_res[0] = %.16f\n",tmp_temps_kernel_res[0]);
	printf("tmp_temps_kernel_res[1] = %.16f\n",tmp_temps_kernel_res[1]);
	printf("tmp_temps_kernel_res[2] = %.16f\n",tmp_temps_kernel_res[2]);
	printf("tmp_temps_kernel_grad[0] = %.16f\n",tmp_temps_kernel_grad[0]);
*/
	double temps2_deriv = omp_get_wtime();

	if(print){
		printf("Milieu :\n");
		printf("-> f = %.16f\n", f);
		std::cin.ignore();
	}


	double temps1_conv = omp_get_wtime();

	float tmp_temps_kernel_regu[9] = {0.,0.,0.,0.,0.,0.,0.,0.,0.};
	regularization<T>(beta_dev, g_dev, b_params, f, indice_x, indice_y, indice_v, M, tmp_temps_kernel_regu);
	temps_kernel[2] += tmp_temps_kernel_regu[7]/1000;//reduction
	temps_kernel[4] += tmp_temps_kernel_regu[0]/1000;//get_gaussian_parameter_maps
	temps_kernel[5] += tmp_temps_kernel_regu[1]/1000;//perform_mirror_effect_before_convolution
	temps_kernel[6] += tmp_temps_kernel_regu[2]/1000;//ConvKernel
	temps_kernel[7] += tmp_temps_kernel_regu[3]/1000;//copy_gpu
	temps_kernel[8] += tmp_temps_kernel_regu[4]/1000;//compute_R_map
	temps_kernel[9] += tmp_temps_kernel_regu[5]/1000;//compute_nabla_R_wrt_theta
	temps_kernel[10] += tmp_temps_kernel_regu[6]/1000;//compute_nabla_R_wrt_m
	temps_kernel[11] += tmp_transfer/1000 + tmp_temps_kernel_regu[8]/1000;;//transfers

	double temps2_conv = omp_get_wtime();

    checkCudaErrors(cudaMemcpy(g, g_dev, n*sizeof(T), cudaMemcpyDeviceToHost));

	if (false)
	{
		lim = n_beta;
		for(int i=0; i<lim; i++){
			printf("g[%d] = %.16f\n",i, g[i]);
		}/*		for(int i=lim-40; i<lim; i++){
			printf("g[%d] = %.16f\n",i, g[i]);
		}*/

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

	free(residual);
	free(std_map_);
	free(b_params);

	checkCudaErrors(cudaFree(beta_dev));
	checkCudaErrors(cudaFree(residual_dev));
	checkCudaErrors(cudaFree(std_map_dev));
    checkCudaErrors(cudaFree(g_dev));
}


template <typename T> 
void f_g_cube_cuda_L_clean_templatized_no_transfers(parameters<T> &M, T& f, T* g_dev, int n, T* beta_dev, int indice_v, int indice_y, int indice_x, T* std_map_dev, T* cube_flattened_dev, double* temps, double temps_transfert_d, double temps_mirroirs, float* temps_kernel)
{

  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  cudaEventRecord(start);

//	dummyInstantiator_sort();
	double temps_conv;
	double temps_deriv;
	double temps_tableaux;
	double temps_copy;
	double temps_f_g_cube;

	float tmp_transfer = 0.;

//	bool print = false;	
	int lim = 100;
/*
    if(indice_x>=256){
        print = true;
    }
*/


//	std::vector<T> b_params(M.n_gauss,0.);
	int i,k,j,l,p;

	int taille_params_flat[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_deriv[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_residual[] = {indice_v, indice_y, indice_x};
	int taille_std_map_[] = {indice_y, indice_x};
	int taille_beta[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_beta_modif[] = {3*M.n_gauss, indice_y, indice_x};
	int taille_cube[] = {indice_v, indice_y, indice_x};

	int product_residual = taille_residual[0]*taille_residual[1]*taille_residual[2];
	int product_params_flat = taille_params_flat[0]*taille_params_flat[1]*taille_params_flat[2];
	int product_deriv = taille_deriv[0]*taille_deriv[1]*taille_deriv[2];
	int product_std_map_ = taille_std_map_[0]*taille_std_map_[1];
	int product_cube = taille_cube[0]*taille_cube[1]*taille_cube[2]; 
	int product_beta = taille_beta[0]*taille_beta[1]*taille_beta[2];
	int product_beta_modif = taille_beta[0]*taille_beta[1]*taille_beta[2];

	size_t size_deriv = product_deriv * sizeof(T);
	size_t size_res = product_residual * sizeof(T);
	size_t size_std = product_std_map_ * sizeof(T);
	size_t size_beta_modif = product_beta_modif * sizeof(T);

	T* b_params = (T*)malloc(M.n_gauss*sizeof(T));

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;

	double temps1_copy = omp_get_wtime();

	f=0.;

//beta est de taille : x,y,3g
//params est de taille : 3g,y,x
//ATTENTION !!!
/*
	for(int i = 0; i<M.n_gauss; i++){
		b_params[i]=beta[n_beta-M.n_gauss+i];
	}
*/

    T* residual_dev = NULL;
    checkCudaErrors(cudaMalloc(&residual_dev, indice_x*indice_y*indice_v*sizeof(T)));
    checkCudaErrors(cudaDeviceSynchronize());
    copy_dev<T><<<ceil(float(indice_x*indice_y*indice_v)/float(BLOCK_SIZE_REDUCTION)),BLOCK_SIZE_REDUCTION>>>(cube_flattened_dev, residual_dev, indice_x*indice_y*indice_v);
    checkCudaErrors(cudaDeviceSynchronize());
//	checkCudaErrors(cudaMemcpy(residual_dev, cube_flattened, indice_x*indice_y*indice_v*sizeof(T), cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMemset(g_dev, 0., n*sizeof(T)));

	temps_copy += omp_get_wtime()-temps1_copy;

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaEventElapsedTime(&tmp_transfer, start, stop));
    checkCudaErrors(cudaDeviceSynchronize());

	


	double temps1_tableaux = omp_get_wtime();
/*
		if(isnan(f)){
			printf("before compute_residual f = %.16f      f=Nan detected !\n",f);
			exit(0);
		}
*/
	float tmp_temps_kernel_res[3] = {0.,0.,0.};
	f =  compute_residual_and_f_less_memory<T>(beta_dev, taille_beta_modif, product_beta, cube_flattened_dev, taille_cube, product_cube, residual_dev, taille_residual, product_residual, std_map_dev, taille_std_map_, product_std_map_, indice_x, indice_y, indice_v, M.n_gauss,tmp_temps_kernel_res);
//	f =  compute_residual_and_f<T>(beta, taille_beta_modif, product_beta, cube_flattened, taille_cube, product_cube, residual, taille_residual, product_residual, std_map_, taille_std_map_, product_std_map_, indice_x, indice_y, indice_v, M.n_gauss);
	double temps2_tableaux = omp_get_wtime();
/*
		if(isnan(f)){
			printf("after compute_residual f = %.16f      f=Nan detected !\n",f);
		T* beta_tmp = NULL;
		beta_tmp = (T*)malloc(n*sizeof(T));
		//int temp = 0;
		checkCudaErrors(cudaMemcpy(beta_tmp, beta_dev, n*sizeof(T), cudaMemcpyDeviceToHost));
	    checkCudaErrors(cudaDeviceSynchronize());

		int compteur = 30;
		printf("print sig maps\n");
		for(int ind = 0; ind < M.n_gauss; ind++){
		for(int ind_y = 0; ind_y < indice_y; ind_y++){
		for(int ind_x = 0; ind_x < indice_x; ind_x++){
			if (compteur ==0) exit(0);
			//printf("%d\n",ind);
			printf("beta_tmp[%d][%d][%d] = %.16f\n",3*ind+2, ind_y, ind_x,beta_tmp[(3*ind+2)*indice_x*indice_y+ind_y*indice_x+ind_x]);
				//checkCudaErrors(cudaMemcpy(x_dev, beta, n*sizeof(T), cudaMemcpyHostToDevice));
			compteur--;
		}
		}
		}
//		printf("print sig maps\n");

		free(beta_tmp);
			exit(0);
		}
*/
	temps_kernel[0] += tmp_temps_kernel_res[0]/1000;//compute_residual
	temps_kernel[1] += tmp_temps_kernel_res[1]/1000;//compute_Q_map
	temps_kernel[2] += tmp_temps_kernel_res[2]/1000;//reduction_loop

//	printf("cpu : %.16f\n",temps2_tableaux-temps1_tableaux);

	double temps1_deriv = omp_get_wtime();

	float tmp_temps_kernel_grad[1]={0.};
	gradient_L_2_beta<T>(g_dev, taille_deriv, product_deriv, beta_dev, taille_beta_modif, product_beta_modif, residual_dev, taille_residual, product_residual, std_map_dev, taille_std_map_, product_std_map_, M.n_gauss, tmp_temps_kernel_grad);
	temps_kernel[3] += tmp_temps_kernel_grad[0]/1000;//compute_nabla_Q

/*
	printf("tmp_temps_kernel_res[0] = %.16f\n",tmp_temps_kernel_res[0]);
	printf("tmp_temps_kernel_res[1] = %.16f\n",tmp_temps_kernel_res[1]);
	printf("tmp_temps_kernel_res[2] = %.16f\n",tmp_temps_kernel_res[2]);
	printf("tmp_temps_kernel_grad[0] = %.16f\n",tmp_temps_kernel_grad[0]);
*/
	double temps2_deriv = omp_get_wtime();

	if(print){
		printf("Milieu :\n");
		printf("-> f = %.16f\n", f);
		std::cin.ignore();
	}


	double temps1_conv = omp_get_wtime();

	float tmp_temps_kernel_regu[9] = {0.,0.,0.,0.,0.,0.,0.,0.,0.};
//	regularization<T>(beta_dev, g_dev, b_params, f, indice_x, indice_y, indice_v, M, tmp_temps_kernel_regu);
	temps_kernel[2] += tmp_temps_kernel_regu[7]/1000;//reduction
	temps_kernel[4] += tmp_temps_kernel_regu[0]/1000;//get_gaussian_parameter_maps
	temps_kernel[5] += tmp_temps_kernel_regu[1]/1000;//perform_mirror_effect_before_convolution
	temps_kernel[6] += tmp_temps_kernel_regu[2]/1000;//ConvKernel
	temps_kernel[7] += tmp_temps_kernel_regu[3]/1000;//copy_gpu
	temps_kernel[8] += tmp_temps_kernel_regu[4]/1000;//compute_R_map
	temps_kernel[9] += tmp_temps_kernel_regu[5]/1000;//compute_nabla_R_wrt_theta
	temps_kernel[10] += tmp_temps_kernel_regu[6]/1000;//compute_nabla_R_wrt_m
	temps_kernel[11] += tmp_transfer/1000 + tmp_temps_kernel_regu[8]/1000;;//transfers

	double temps2_conv = omp_get_wtime();

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
	free(b_params);

	checkCudaErrors(cudaFree(residual_dev));
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
/*
// find center position of kernel (half of kernel size)
kCenterX = 1;
kCenterY = 1;
int rows = dim_y+4;
int cols = dim_x+4;
int kRows = 3;
int kCols = 3;
for(int i=0; i < rows; ++i)              // rows
{
    for(int j=0; j < cols; ++j)          // columns
    {
        for(int m=0; m < kRows; ++m)     // kernel rows
        {
            int mm = kRows - 1 - m;      // row index of flipped kernel

            for(int n=0; n < kCols; ++n) // kernel columns
            {
                int nn = kCols - 1 - n;  // column index of flipped kernel

                // index of input signal, used for checking boundary
                ii = i + (kCenterY - mm);
                jj = j + (kCenterX - nn);

                // ignore input samples which are out of bound
                if( ii >= 0 && ii < rows && jj >= 0 && jj < cols )
                    ext_conv[i][j] += extended[ii][jj] * kernel[mm*3+nn];
            }
        }
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

					if( ii >= 1 && ii <= dim_y+4 && jj>=1 && jj<= dim_x+4 )
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

					if( ii >= 1 && ii <= dim_y+4 && jj>=1 && jj<= dim_x+4 )
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
void one_D_to_three_D_inverted_dimensions(T* vector, std::vector<std::vector<std::vector<T>>> &cube_3D, int dim_x, int dim_y, int dim_v)
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
	            cube_3D[k][j][i] = vector[i*dim_y*dim_x+j*dim_x+k];
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
void three_D_to_one_D_inverted_dimensions(const std::vector<std::vector<std::vector<T>>> &cube_3D, T* vector, int dim_x, int dim_y, int dim_v)
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
	            vector[i*dim_y*dim_x+j*dim_x+k] = cube_3D[k][j][i];
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

template void f_g_cube_cuda_L_clean_templatized_no_transfers(parameters<double> &, double&, double*, int, double*, int, int, int, double*, double*, double*, double, double, float* );
template void f_g_cube_cuda_L_clean_templatized(parameters<double> &, double&, double*, int, double*, int, int, int, std::vector<std::vector<double>> &, double*, double*, double, double, float* );
template void f_g_cube_cuda_L_clean_templatized_less_transfers(parameters<double> &, double&, double*, int, double*, int, int, int, std::vector<std::vector<double>> &, double*, double*, double, double, float* );

template double model_function(int x, double a, double m, double s);
template void f_g_cube_fast_unidimensional(parameters<double>&, double&, double*, int, double*, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, double*, double*);	
template void f_g_cube_fast_unidimensional_test(parameters<double>&, double&, double*, int, double*, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, double*, double*);	
template double f_g_cube_fast_unidimensional_return(parameters<double>&, double*, int, double*, std::vector<std::vector<std::vector<double>>>&, double*, int, int, int, double*, double*);	

template void convolution_2D_mirror_flat(const parameters<double> &M, double* image, double* conv, int dim_y, int dim_x, int dim_k);
template void convolution_2D_mirror(const parameters<double> &M, const std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &conv, int dim_y, int dim_x, int dim_k);

template void f_g_cube_fast_clean(parameters<double>&, double &, double*, int n, std::vector<std::vector<std::vector<double>>>&, double*, int , int , int , std::vector<std::vector<double>>&, double*);
template void f_g_cube_not_very_fast_clean(parameters<double>&, double &, double*, int n, std::vector<std::vector<std::vector<double>>>&, double*, int , int , int , std::vector<std::vector<double>>&, double*);
template void f_g_cube_fast_clean_optim_CPU_lib(parameters<double>&, double &, double*, int n, std::vector<std::vector<std::vector<double>>>&, double*, int , int , int , std::vector<std::vector<double>>&, double**, double*);

template void one_D_to_three_D_same_dimensions(double*, std::vector<std::vector<std::vector<double>>>&, int, int, int);
template void one_D_to_three_D_inverted_dimensions(double*, std::vector<std::vector<std::vector<double>>>&, int, int, int);
template void three_D_to_one_D_same_dimensions(const std::vector<std::vector<std::vector<double>>>&, double*, int, int, int);
template void three_D_to_one_D_inverted_dimensions(const std::vector<std::vector<std::vector<double>>>&, double*, int, int, int);

template double myfunc_spec(std::vector<double>&);
template void myresidual(std::vector<double>&, std::vector<double>&, std::vector<double>&, int);





template void f_g_cube_cuda_L_clean_templatized_no_transfers(parameters<float> &, float&, float*, int, float*, int, int, int, float*, float*, double*, double, double, float* );
template void f_g_cube_cuda_L_clean_templatized(parameters<float> &, float&, float*, int, float*, int, int, int, std::vector<std::vector<float>> &, float*, double*, double, double, float* );
template void f_g_cube_cuda_L_clean_templatized_less_transfers(parameters<float> &, float&, float*, int, float*, int, int, int, std::vector<std::vector<float>> &, float*, double*, double, double, float* );

template float model_function(int x, float a, float m, float s);
template void f_g_cube_fast_unidimensional(parameters<float>&, float&, float*, int, float*, std::vector<std::vector<std::vector<float>>>&, float*, int, int, int, float*, double*);	
template void f_g_cube_fast_unidimensional_test(parameters<float>&, float&, float*, int, float*, std::vector<std::vector<std::vector<float>>>&, float*, int, int, int, float*, double*);	
template float f_g_cube_fast_unidimensional_return(parameters<float>&, float*, int, float*, std::vector<std::vector<std::vector<float>>>&, float*, int, int, int, float*, double*);	

template void convolution_2D_mirror_flat(const parameters<float> &M, float* image, float* conv, int dim_y, int dim_x, int dim_k);
template void convolution_2D_mirror(const parameters<float> &M, const std::vector<std::vector<float>> &image, std::vector<std::vector<float>> &conv, int dim_y, int dim_x, int dim_k);

template void f_g_cube_fast_clean(parameters<float>&, float &, float*, int n, std::vector<std::vector<std::vector<float>>>&, float*, int , int , int , std::vector<std::vector<float>>&, double*);
template void f_g_cube_not_very_fast_clean(parameters<float>&, float &, float*, int n, std::vector<std::vector<std::vector<float>>>&, float*, int , int , int , std::vector<std::vector<float>>&, double*);
template void f_g_cube_fast_clean_optim_CPU_lib(parameters<float>&, float &, float*, int n, std::vector<std::vector<std::vector<float>>>&, float*, int , int , int , std::vector<std::vector<float>>&, float**, double*);

template void one_D_to_three_D_same_dimensions(float*, std::vector<std::vector<std::vector<float>>>&, int, int, int);
template void one_D_to_three_D_inverted_dimensions(float*, std::vector<std::vector<std::vector<float>>>&, int, int, int);
template void three_D_to_one_D_same_dimensions(const std::vector<std::vector<std::vector<float>>>&, float*, int, int, int);
template void three_D_to_one_D_inverted_dimensions(const std::vector<std::vector<std::vector<float>>>&, float*, int, int, int);

template float myfunc_spec(std::vector<float>&);
template void myresidual(std::vector<float>&, std::vector<float>&, std::vector<float>&, int);



