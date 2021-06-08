#include "f_g_cube_vecto.hpp"

#define print false

void f_g_cube_fast_unidimensional_no_template(parameters<double> &M, double &f, double* __restrict__ g, int n, double* __restrict__ cube, std::vector<std::vector<std::vector<double>>>& __restrict__ cube_for_cache, double* __restrict__ beta, int indice_v, int indice_y, int indice_x, double* __restrict__ std_map, double* temps){	
//void f_g_cube_fast_unidimensional(parameters<double> &M, double &f, double* g, int n, double* cube, std::vector<std::vector<std::vector<double>>>& cube_for_cache, double* beta, int indice_v, int indice_y, int indice_x, double* std_map, double* temps){

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

    double Kernel[9] = {0,-0.25,0,-0.25,1,-0.25,0,-0.25,0};

	std::vector<double> b_params(M.n_gauss,0.);
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

	size_t size_res = product_residual * sizeof(double);
	size_t size_std = product_std_map_ * sizeof(double);
	size_t size_image_conv = product_image_conv * sizeof(double);

	double* residual = (double*)__builtin_malloc(size_res);
	double* std_map_ = (double*)__builtin_malloc(size_std);

	double* conv_amp = (double*)__builtin_malloc(size_image_conv);
	double* conv_mu = (double*)__builtin_malloc(size_image_conv);
	double* conv_sig = (double*)__builtin_malloc(size_image_conv);
	double* conv_conv_amp = (double*)__builtin_malloc(size_image_conv);
	double* conv_conv_mu = (double*)__builtin_malloc(size_image_conv);
	double* conv_conv_sig = (double*)__builtin_malloc(size_image_conv);
	double* image_amp = (double*)__builtin_malloc(size_image_conv);
	double* image_mu = (double*)__builtin_malloc(size_image_conv);
	double* image_sig = (double*)__builtin_malloc(size_image_conv);

	int n_beta = (3*M.n_gauss*indice_x*indice_y)+M.n_gauss;

	double temps1_ravel = omp_get_wtime();
	for(int i=0; i<indice_y; i++){
		#pragma omp simd
		for(int j=0; j<indice_x; j++){
			std_map_[j*indice_y+i] = std_map[i*indice_x+j];
		}
	}
	#pragma omp simd
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


//////	double* hypercube_tilde = NULL;
//////	hypercube_tilde = (double*)malloc(indice_x*indice_y*indice_v*sizeof(double));

	double* par = NULL;
	par = (double*)__builtin_malloc(3*sizeof(double));

//	omp_set_num_threads(50);
///	omp_set_num_threads(1);
///	#pragma omp parallel private(j,i) shared(hypercube_tilde, residual ,par,M,beta,std_map_,indice_v,indice_y,indice_x)
///	{
//	printf("num threads = %d\n", omp_get_num_threads());
//	#pragma omp for
//	#pragma nosimd
//		#pragma simd
	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
			for (int p=0; p<indice_v; p++){
				double spec = double(p+1); //+1
				double accu = 0.;
				#pragma omp simd reduction(+: accu)
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
///	}
//////	free(hypercube_tilde);
	free(par);

	#pragma omp simd
	for(j=0; j<indice_x; j++){
		for(i=0; i<indice_y; i++){
			if(std_map_[j*indice_y+i]>0.){
				double accu = 0.;
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

	double temps1_conv = omp_get_wtime();
	for(int i=0; i<M.n_gauss; i++){
		#pragma omp simd
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
		convolution_2D_mirror_flat_vecto(M, image_amp, conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror_flat_vecto(M, image_mu, conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror_flat_vecto(M, image_sig, conv_sig, indice_y, indice_x,3);
		convolution_2D_mirror_flat_vecto(M, conv_amp, conv_conv_amp, indice_y, indice_x,3);
		convolution_2D_mirror_flat_vecto(M, conv_mu, conv_conv_mu, indice_y, indice_x,3);
		convolution_2D_mirror_flat_vecto(M, conv_sig, conv_conv_sig, indice_y, indice_x,3);

		#pragma omp simd
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
///	#pragma omp parallel private(j,l) shared(g,M,beta,std_map_,residual,indice_v,indice_y,indice_x)
///	{
///	#pragma omp for
//	#pragma omp simd						
	for(l=0; l<indice_x; l++){
		for(j=0; j<indice_y; j++){
			if(std_map_[l*indice_y+j]>0.){
				for(int i=0; i<M.n_gauss; i++){
					double par0 = beta[l*indice_y*3*M.n_gauss +j*3*M.n_gauss +(0+3*i)];
					double par1_ = beta[l*indice_y*3*M.n_gauss +j*3*M.n_gauss +(1+3*i)];
					double par2 = beta[l*indice_y*3*M.n_gauss +j*3*M.n_gauss +(2+3*i)];	
					#pragma omp simd						
					for(int k=0; k<indice_v; k++){	
						double par1 = double(k+1) - par1_;//+1   double(p+1)
						g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(0+3*i)] += exp(-pow( par1,2.)/(2*pow(par2,2.)) )*residual[l*indice_y*indice_v + j*indice_v + k]/pow(std_map_[l*indice_y+j],2);
						g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(1+3*i)] += par0*par1/pow(par2,2.) * exp(-pow(par1,2.)/(2*pow(par2,2.)) )*residual[l*indice_y*indice_v + j*indice_v + k]/pow(std_map_[l*indice_y+j],2);
						g[l*indice_y*3*M.n_gauss + j*3*M.n_gauss +(2+3*i)] += par0*pow(par1, 2.)/(pow(par2,3.)) * exp(-pow(par1,2.)/(2*pow(par2,2.)) )*residual[l*indice_y*indice_v + j*indice_v + k]/pow(std_map_[l*indice_y+j],2);
				}
			}
		}
	}
///	}
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

void convolution_2D_mirror_flat_vecto(const parameters<double> &M, double* __restrict__ image, double* __restrict__ conv, int dim_y, int dim_x, int dim_k)
{
	double kernel[]= {0.,-0.25,0.,-0.25,1.,-0.25,0.,-0.25,0.}; 
	int ii(0),jj(0),mm(0),nn(0),kCenterY(0), kCenterX(0);

	std::vector<std::vector<double>> ext_conv(dim_y+4, std::vector<double>(dim_x+4,0.));
	std::vector<std::vector<double>> extended(dim_y+4, std::vector<double>(dim_x+4,0.));


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

