#include "callback_cpu.h"

void callback_test(parameters &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int* dim, std::vector<std::vector<double>> &std_map, double** assist_buffer){

    int indice_x = dim[2];
    int indice_y = dim[1];
    int indice_v = dim[0];

	std::vector<std::vector<std::vector<double>>> deriv(3*M.n_gauss,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
	std::vector<std::vector<std::vector<double>>> residual(indice_v,std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_x,0.)));
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

/*	for(int i = 0; i<n_beta; i++){
		printf("beta[%d] = %f\n",i,beta[i]);
	}
*/
	one_D_to_three_D_same_dimensions(beta, params, 3*M.n_gauss, indice_y, indice_x);

	for(int i=0; i<indice_y; i++){
		for(int j=0; j<indice_x; j++){
			std::vector<double> residual_1D(indice_v,0.);
			std::vector<double> params_flat(params.size(),0.);
			std::vector<double> cube_flat(cube[0][0].size(),0.);

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

	int i;

//	#pragma omp parallel private(i) shared(params,deriv,std_map,residual,indice_v,indice_y,indice_x)
//	{
//	#pragma omp for
	for(i=0; i<M.n_gauss; i++){
		for(int k=0; k<indice_v; k++){
			for(int j=0; j<indice_y; j++){
				for(int l=0; l<indice_x; l++){
					if(std_map[j][l]>0.){
						double spec = double(k+1);
						deriv[0+3*i][j][l] += exp(-pow( spec-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*residual[k][j][l]/pow(std_map[j][l],2);
						deriv[1+3*i][j][l] += params[3*i][j][l]*( spec - params[1+3*i][j][l])/pow(params[2+3*i][j][l],2.) * exp(-pow( spec-params[1+3*i][j][l],2.)/	(2*pow(params[2+3*i][j][l],2.)) )*residual[k][j][l]/pow(std_map[j][l],2);
						deriv[2+3*i][j][l] += params[3*i][j][l]*pow( spec - params[1+3*i][j][l], 2.)/(pow(params[2+3*i][j][l],3.)) * exp(-pow( spec-params[1+3*i][j][l],2.)/(2*pow(params[2+3*i][j][l],2.)) )*residual[k][j][l]/pow(std_map[j][l],2);
					}
				}
			}
		}
	}
//	}

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

    three_D_to_one_D_same_dimensions(deriv, g, 3*M.n_gauss, indice_y, indice_x);

	}

void three_D_to_one_D_same_dimensions(const std::vector<std::vector<std::vector<double>>> &cube_3D, double* vector, int dim_x, int dim_y, int dim_v)
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


void one_D_to_three_D_same_dimensions(double* vector, std::vector<std::vector<std::vector<double>>> &cube_3D, int dim_x, int dim_y, int dim_v)
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



void myresidual(double params[], double line[], std::vector<double> &residual, int n_gauss_i) {
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

void myresidual(std::vector<double> &params, std::vector<double> &line, std::vector<double> &residual, int n_gauss_i) {
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

double model_function(int x, double a, double m, double s) {

	return a*exp(-pow((double(x)-m),2.) / (2.*pow(s,2.)));

}

void convolution_2D_mirror(const parameters &M, const std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &conv, int dim_y, int dim_x, int dim_k)
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

double myfunc_spec(std::vector<double> &residual) {
	double S(0.);
	for(int p(0); p<residual.size(); p++) {
		S+=pow(residual[p],2);
	}
	return 0.5*S;
}
