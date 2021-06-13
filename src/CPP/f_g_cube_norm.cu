#include "f_g_cube_norm.hpp"
//#include "f_g_cube_gpu.hpp"
#include "kernels_for_hybrid.cu"
#include "kernels_for_hybrid_norm.cu"

#define print false

template <typename T> 
void f_g_cube_cuda_L_clean_templatized_norm(parameters<T> &M, T& f, T* g, int n, T* beta, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened, double* temps, double temps_transfert_d, double temps_mirroirs, float* temps_kernel)
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
	regularization_norm<T>(beta_dev, g_dev, b_params, f, indice_x, indice_y, indice_v, M, tmp_temps_kernel_regu);
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
void f_g_cube_cuda_L_clean_templatized_less_transfers_norm(parameters<T> &M, T& f, T* g, int n, T* beta, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened_dev, double* temps, double temps_transfert_d, double temps_mirroirs, float* temps_kernel)
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
	regularization_norm<T>(beta_dev, g_dev, b_params, f, indice_x, indice_y, indice_v, M, tmp_temps_kernel_regu);
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



template void f_g_cube_cuda_L_clean_templatized_norm(parameters<double> &, double&, double*, int, double*, int, int, int, std::vector<std::vector<double>> &, double*, double*, double, double, float* );
template void f_g_cube_cuda_L_clean_templatized_less_transfers_norm(parameters<double> &, double&, double*, int, double*, int, int, int, std::vector<std::vector<double>> &, double*, double*, double, double, float* );


template void f_g_cube_cuda_L_clean_templatized_norm(parameters<float> &, float&, float*, int, float*, int, int, int, std::vector<std::vector<float>> &, float*, double*, double, double, float* );
template void f_g_cube_cuda_L_clean_templatized_less_transfers_norm(parameters<float> &, float&, float*, int, float*, int, int, int, std::vector<std::vector<float>> &, float*, double*, double, double, float* );
