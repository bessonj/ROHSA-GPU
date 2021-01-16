#ifndef DEF_ALGO_ROHSA
#define DEF_ALGO_ROHSA

#include "../L-BFGS-B-C/src/lbfgsb.h"
#include "hypercube.hpp"
#include "parameters.hpp"
#include <iostream>
#include <stdio.h>
#include <cmath>
//#include <math.h>
#include <string>
#include <fstream>
#include <valarray>
#include <CCfits/CCfits>
#include <vector>
#include <omp.h>

#include <array>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
#include "gradient.hpp"
#include "convolutions.hpp"
#include "f_g_cube_gpu.hpp"
#include "culbfgsb.h"
//#include "callback_cpu.h"
#include "f_g_cube.hpp"

#define print false

/**
 * @brief This class concerns the ROHSA algorithm and the optimization algorithm.
 *
 *
 * The section below presents the attributes of this class.
 * @param grid_params is a 3D array containing the gaussian parameters \f$\lambda, \mu, \sigma \f$ depending on the spatial position. It is a \f$ 3 n\_gauss \times dim\_y \times dim\_x \f$ dimensional array. 
 * @param fit_params is similar to grid_params (gaussian parameters) but its dimensions varies through multiresolution. It is a \f$ 3 n\_gauss \times 2^k \times 2^k \f$ for \f$ 0 < k < n\_side \f$ dimensional array.
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 *
 */

// mettre des const à la fin des déclarations si on ne modifie pas l'objet / les attributs

template <typename T>
class algo_rohsa
{
	public:

	algo_rohsa(parameters<T> &M, hypercube<T> &Hypercube); //constructeur

/**
 * @brief Each iteration into the main loop corresponds to a resolution. At each iteration, we compute a piecewise spatially averaged array from the data extracted from the FITS file, then we compute the standard deviation map along each spatial position, we set the upper and lower boundaries and we solve iteratively the problem using L-BFGS-B-C, finally, we project the result onto a higher resolution grid.      
 * 
 * The section below presents the attributes of this class.
 * @param M is an object, its attributes are the variables whose values are chosen by the user in parameters.txt, these are variables related to the data structure, the gaussian model and the black box of L-BFGS-B-C used in the minimize() function.
 * @param Hypercube is an object containing the attributes data and cube. "data" is the array extracted from the FITS file and truncated because of the unexploitable spectral ranges of the data ; "cube" is an array whose side is a power of 2 and containing the data array.
 * 
 * 
 * Since we don't want to make useless computations, we use cube (and fit_params) until the last level of multiresolution. When we reach the last level, we use data (and grid_params) which has the original dimensions of the hypercube (after truncation on the spectral dimension).  
 * 
 * 
 */

	void descente(parameters<T> &M, std::vector<std::vector<std::vector<T>>> &grid_params, std::vector<std::vector<std::vector<T>>> &fit_params); //!< main loop for the multiresolution process

//	void descente(parameters<T> &M, std::vector<std::vector<std::vector<T>>> &grid_params, std::vector<std::vector<std::vector<T>>> &fit_params); //!< main loop for the multiresolution process
/**
 * @brief Similar to descente() but without regularization.
 * 
 */
	void ravel_2D(const std::vector<std::vector<T>> &map, std::vector<T> &vector, int dim_y, int dim_x);
	void ravel_3D(const std::vector<std::vector<std::vector<T>>> &cube_3D, std::vector<T> &vector, int dim_v, int dim_y, int dim_x);

	void init_bounds(parameters<T> &M, std::vector<T>& line, int n_gauss_local, std::vector<T> &lb, std::vector<T> &ub, bool _init);
	void init_bounds_double(parameters<T> &M, std::vector<double>& line, int n_gauss_local, std::vector<double>& lb, std::vector<double>& ub, bool _init);

	void mean_array(int power, std::vector<std::vector<std::vector<T>>> &cube_mean);

	void init_spectrum(parameters<T> &M, std::vector<double> &line, std::vector<double> &params);	
//	void init_spectrum(parameters<T> &M, std::vector<T> &line, std::vector<T> &params); //!< Initializes spectrum (called during first iteration) 

	T model_function(int x, T a, T m, T s);

	int minloc(std::vector<T> &tab);
	int minloc_double(std::vector<double> &tab);
	void minimize_spec(parameters<T> &M, long n, long m, std::vector<double> &x_v, std::vector<double> &lb_v, int n_gauss_i, std::vector<double> &ub_v, std::vector<double> &line_v);
	//void minimize_spec(parameters<T> &M, long n, long m, std::vector<T> &x_v, std::vector<T> &lb_v, int n_gauss_i,std::vector<T> &ub_v, std::vector<T> &line_v); //!< Solves the optimization problem during the first iteration, it calls the L-BFGS-B black box.
	void minimize_spec_save(parameters<T> &M, long n, long m, std::vector<T> &x_v, std::vector<T> &lb_v, int n_gauss_i,std::vector<T> &ub_v, std::vector<T> &line_v); //!< Solves the optimization problem during the first iteration, it calls the L-BFGS-B black box.

/**
 * @brief Routine solving the optimization problem.        
 * 
 * Solves the optimization problem during one resolution level of multiresolution, it calls the L-BFGS-B black box.
 *
 * @param beta is the flattened gaussian parameters array. It contains the maps of \f$ \lambda, \mu, \sigma \f$ and also \f$ m  \f$, inserted in the last n_gauss indices of the array. Its length is therefore \f$ n\_beta = 3n\_gauss \times dim\_y \times dim\_x + n\_gauss \f$. We may write **beta** as \f$ ( \theta, m ) \f$, where \f$ \theta = \left ( \begin{bmatrix} 
\lambda_1   \\ 
\mu_1 \\
\sigma_1
\end{bmatrix} , 
\begin{bmatrix} 
\lambda_2   \\ 
\mu_2 \\
\sigma_2
\end{bmatrix}, ... , 
\begin{bmatrix} 
\lambda_{n\_gauss}   \\ 
\mu_{n\_gauss} \\
\sigma_{n\_gauss}
\end{bmatrix} \right)
\f$ is a line vector of maps representing the gaussian parameters of each gaussian.
 * 
 * @return **beta** \f$ \equiv ( \theta, m ) \f$ such as it solves the optimization problem below.
 *
 *  \f[
\begin{align*}
\text{Minimize }
J( \theta, m ) = \frac{1}{2} \sum_{\nu_z,\textbf{r}} \left( \frac{residual(\nu_z,\textbf{r})}{std\_map(\textbf{r })}\right)^2 &+ \frac{1}{2} \sum_{n=1}^{n\_gauss} ( \lambda_a ||\textbf{D} a_n||_{L^2}^2 \\
 &+ \lambda_\mu ||\textbf{D} \mu_n||_{L^2}^2 \\
 &+ \lambda_\sigma ||\textbf{D} \sigma_n||_{L^2}^2 \\
 &+ \lambda_\sigma' ||\sigma_n - m_n||_{L^2}^2 
)
\end{align*}

  \f] 
 *
 * Where :  
 *
 * \f$ residual \f$ is the difference between the model (sum of gaussians) and the cube. It is computed though the spatial and the spectral dimensions. 
 *
 * **D** is a discrete derivative. 
 *
 * \f$ \lambda_a, \lambda_\mu, \lambda_\sigma \f$ and \f$ \lambda_\sigma' \f$ are constant hyperparameters stored as attributes of the object \f$ M \f$. 
 *
 * \f$ n\_gauss \f$ is the number of gaussians. 
 *
 * \f$ \textbf{r} \f$ represents spatial coordinates. 
 *
 * \f$ \nu_z \f$ is the spectral dimension. 
 *
 * \f$ a_n, \mu_n \f$ and \f$ \sigma_n \f$ are maps of the gaussian parameters of the n-th gaussian. 
 * 
 * 
 * The smoothness of the solution results from the terms (aligned to the right of the "+" sign) involving the convolutions. \f$ R(\theta,m) \f$ denotes these terms (\f$ R\f$ for regularization).
 * 
 * 
 * 
 * 
 * L-BFGS-B approximates the solution with :
 * 
   \f[
	\begin{align*}
	\left\{
	\begin{array}{ll}
	\theta_{k+1} = \theta_k - \alpha_k \textbf{H}_k^{-1} \nabla J(\theta_k, m_k)\\
	\theta_0 = \theta_{initialization}
	\end{array}
	\right. \label{formulas}
	\end{align*}
    \f]
 * where \f$ \textbf{H}_k \f$ is a matrix and  \f$  \alpha_k  \f$ a real number.
 * It stops when the algorithm reaches a maximum of iterations or if the projected gradient is small enough.
 * 
 * As we can see we need to compute the gradient of \f$ J \f$. 
 * 
     \f[
\nabla J(\theta, m) = 
\begin{bmatrix} 
\nabla_{\theta} residual(\theta) \times \frac{residual(\theta)}{(std\_map(\textbf{r }))^2}   \\ 
0
\end{bmatrix}
+
\begin{bmatrix} 

\nabla_{\theta} R(\theta,m)\\ 
\nabla_{m} R(\theta,m)
\end{bmatrix}
     \f]
 * 
 * 
 * \f$ \nabla residual(\theta) \f$ is given by its litteral expression :
     \f[
\nabla_{\theta} residual(\nu_z,\theta(\bf{r}))= \begin{bmatrix} 
exp\left( -\frac{(\nu_z-\mu_1(\bf{r}))^2}{2\sigma_1(\bf{r})^2} \right) \\ 

\frac{a_1(\nu_z-\mu_1(\bf{r}))}{\sigma_1(\bf{r})^2} exp\left( -\frac{(\nu_z-\mu_1(\bf{r}))^2}{2\sigma_1(\bf{r})^2} \right) \\

\frac{a_1(\nu_z-\mu_1(\bf{r}))^2}{\sigma_1(\bf{r})^3} exp\left( -\frac{(\nu_z-\mu_1(\bf{r}))^2}{2\sigma_1(\bf{r})^2} \right) \\

\vdots\\

exp\left( -\frac{(\nu_z-\mu_{n\_gauss}(\bf{r}))^2}{2\sigma_{n\_gauss}(\bf{r})^2} \right) \\ 

\frac{a_{n\_gauss}(\nu_z-\mu_{n\_gauss}(\bf{r}))}{\sigma_{n\_gauss}(\bf{r})^2} exp\left( -\frac{(\nu_z-\mu_{n\_gauss}(\bf{r}))^2}{2\sigma_{n\_gauss}(\bf{r})^2} \right) \\

\frac{a_{n\_gauss}(\nu_z-\mu_{n\_gauss}(\bf{r}))^2}{\sigma_{n\_gauss}(\bf{r})^3} exp\left( -\frac{(\nu_z-\mu_{n\_gauss}(\bf{r}))^2}{2\sigma_{n\_gauss}(\bf{r})^2} \right)
\end{bmatrix}

     \f]
 
 *   
 *   
 *    \f$ \nabla R(\theta,m) \f$  can be also expressed through a litteral expression :
 \f[
\nabla_{\theta} R(\theta, m)= \begin{bmatrix} 
\lambda_a D^t D a_1 \\ 
\lambda_\mu D^t D \mu_1 \\ 
\lambda_\sigma D^t D \sigma_1 \\ 
\lambda'_\sigma(\sigma_1-m_1)\\
\vdots\\
\lambda'_\sigma(\sigma_{n\_gauss}-m_{n\_gauss})
\end{bmatrix}

     \f]
 *   
 *   
 \f[
\nabla_{m} R(\theta, m)= \begin{bmatrix} 
-\sum_r \lambda'_\sigma(\sigma_1-m_1) \\ 
\vdots\\
-\sum_r \lambda'_\sigma(\sigma_{n\_gauss}-m_{n\_gauss})
\end{bmatrix}

     \f]

 *   
 *  
 * The function **setulb()** used as a black box corresponds to the computation of \f$ \theta_{k+1} \f$, the Hessian is approached using the M.m (set in parameters.txt) lasts results (hence the "L" in "L-BFGS" which stands for "limited memory"). The function **f_g_cube()** computes the gradient \f$ g = \nabla J(\theta, m) \f$ and a scalar named \f$ f \f$.
 * 
 * 
 * 
 * 
 */
 
	void myresidual(T* params, T* line, std::vector<T> &residual, int n_gauss_i);
	void myresidual(std::vector<T> &params, std::vector<T> &line, std::vector<T> &residual, int n_gauss_i);
	void myresidual_double(double* params, double* line, std::vector<double> &residual, int n_gauss_i);

	T myfunc_spec(std::vector<T> &residual);
	double myfunc_spec_double(std::vector<double> &residual);

	void mygrad_spec(T* gradient, std::vector<T> &residual, T* params, int n_gauss_i);
	void mygrad_spec_double(double* gradient, std::vector<double> &residual, double* params, int n_gauss_i);

	void upgrade(parameters<T> &M, std::vector<std::vector<std::vector<T>>> &cube, std::vector<std::vector<std::vector<T>>> &params, int power);

	void go_up_level(std::vector<std::vector<std::vector<T>>> &fit_params);

	void set_stdmap(std::vector<std::vector<T>> &std_map, std::vector<std::vector<std::vector<T>>> &cube_or_data, int lb, int ub);
//	void set_stdmap(std::vector<std::vector<T>> &std_map, std::vector<std::vector<std::vector<T>>> &cube, int lb, int ub); //!< Computes the standard deviation map for every spatial position.
	void set_stdmap_transpose(std::vector<std::vector<T>> &std_map, std::vector<std::vector<std::vector<T>>> &cube_or_data, int lb, int ub);
	
/**
 * @brief We prepare the boundary conditions and we call the routine that will make the model fit the data.        
 * 
 * @return 
 * **params** and **b_params** : 3D array of gaussian parameters and a 1D array representing an additionnal term of the cost function.
 * 
 * The section below presents the attributes of this class.
 * 
 * @param cube_avgd_or_data is either the averaged cube array corresponding to a level of multiresolution or the data array (last level of multiresolution). 
 * @param M is an object, its attributes are the variables whose values are chosen by the user in parameters.txt, these are variables related to the data structure, the gaussian model and the black box of L-BFGS-B-C used in the minimize() function.
 * @param std_map is the standard deviation map computed from the cube
 * 
 *
 * 
 * Using init_bounds() we prepare the upper and lower boundary conditions ub and lb.
 * Every array is flattened. **We use "beta" as the flattened gaussian parameters array.**
 * (Remark : We can afford these transformations because we haven't yet reached the optimization loop of minimize() which calls L-BFGS-B-C.) 
 * 
 * 
 */
	void update_clean(parameters<T> &M, std::vector<std::vector<std::vector<T>>> &cube_avgd_or_data, std::vector<std::vector<std::vector<T>>> &params, std::vector<std::vector<T>> &std_map, int indice_x, int indice_y, int indice_v,std::vector<T> &b_params);//!< Prepares boundary conditions and calls the minimize function.

	void minimize_clean(parameters<double> &M, long n, long m, double* beta, double* lb, double* ub, std::vector<std::vector<std::vector<double>>> &cube, double* std_map_, int dim_x, int dim_y, int dim_v, double* cube_flattened);
	void minimize_clean_cpu(parameters<T> &M, long n, long m, T* beta, T* lb, T* ub, std::vector<std::vector<std::vector<T>>> &cube, std::vector<std::vector<T>> &std_map, int dim_x, int dim_y, int dim_v, T* cube_flattened);
	void minimize_clean_gpu(parameters<T> &M, long n, long m, T* beta, T* lb, T* ub, std::vector<std::vector<std::vector<T>>> &cube, std::vector<std::vector<T>> &std_map, int dim_x, int dim_y, int dim_v, T* cube_flattened);
	

	void reshape_down(std::vector<std::vector<std::vector<T>>> &tab1, std::vector<std::vector<std::vector<T>>>&tab2);


	void initialize_array(T* array, int size, T value);
	void initialize_array_double(double* array, int size, double value);
	void three_D_to_one_D(const std::vector<std::vector<std::vector<T>>> &cube_3D, std::vector<T> &vector, int dim_x, int dim_y, int dim_v);
	void three_D_to_one_D(const std::vector<std::vector<std::vector<T>>> &cube_3D, T* vector, int dim_x, int dim_y, int dim_v);
	void one_D_to_three_D_inverted_dimensions(T* vector, std::vector<std::vector<std::vector<T>>> &cube_3D, int dim_x, int dim_y, int dim_v);
	void one_D_to_three_D_same_dimensions(T* vector, std::vector<std::vector<std::vector<T>>> &cube_3D, int dim_x, int dim_y, int dim_v);
	void three_D_to_one_D_inverted_dimensions(const std::vector<std::vector<std::vector<T>>> &cube_3D, T* vector, int dim_x, int dim_y, int dim_v);;
	void three_D_to_one_D_same_dimensions(const std::vector<std::vector<std::vector<T>>> &cube_3D, T* vector, int dim_x, int dim_y, int dim_v);;
	void one_D_to_three_D_inverted_dimensions_double(double* vector, std::vector<std::vector<std::vector<T>>> &cube_3D, int dim_x, int dim_y, int dim_v);
	void one_D_to_three_D_same_dimensions_double(double* vector, std::vector<std::vector<std::vector<T>>> &cube_3D, int dim_x, int dim_y, int dim_v);
	void three_D_to_one_D_inverted_dimensions_double(const std::vector<std::vector<std::vector<T>>> &cube_3D, double* vector, int dim_x, int dim_y, int dim_v);;
	void three_D_to_one_D_same_dimensions_double(const std::vector<std::vector<std::vector<T>>> &cube_3D, double* vector, int dim_x, int dim_y, int dim_v);;
	void ravel_3D(const std::vector<std::vector<std::vector<T>>> &cube, T* vector, int dim_v, int dim_y, int dim_x);
	void unravel_3D(const std::vector<T> &vector, std::vector<std::vector<std::vector<T>>> &cube, int dim_v, int dim_y, int dim_x);
	void unravel_3D(T* vector, std::vector<std::vector<std::vector<T>>> &cube, int dim_v, int dim_y, int dim_x);
	void unravel_3D_T(T* vector, std::vector<std::vector<std::vector<T>>> &cube, int dim_x, int dim_y, int dim_z);
	T mean(const std::vector<T> &array);
	T Std(const std::vector<T> &array);
	T std_2D(const std::vector<std::vector<T>> &map, int dim_y, int dim_x);
	T max_2D(const std::vector<std::vector<T>> &map, int dim_y, int dim_x);
	T mean_2D(const std::vector<std::vector<T>> &map, int dim_y, int dim_x);
	void std_spectrum(int dim_x, int dim_y, int dim_v);
	void mean_spectrum(int dim_x, int dim_y, int dim_v);
	void max_spectrum(int dim_x, int dim_y, int dim_v);
	void max_spectrum_norm(int dim_x, int dim_y, int dim_v, T norm_value);
	void mean_parameters(std::vector<std::vector<std::vector<T>>> &params);

	std::vector<std::vector<std::vector<T>>> grid_params; //!< 3D array containing the gaussian parameters \f$\lambda, \mu, \sigma \f$ depending on the spatial position. Dimensions : It is a \f$ 3 n\_gauss \times dim\_y \times dim\_x \f$.
	std::vector<std::vector<std::vector<T>>> fit_params; //!< same as grid_params (gaussian parameters) but this array is used through multiresolution. Dimensions : \f$ 3 n\_gauss \times 2^k \times 2^k \f$ for \f$ 0 < k < n\_side \f$.

	T* cube_or_dat_flattened; //!< Data flattened at each iteration. Because of the multiresolution process, it has to be computed at each iteration.

//	Computationnal tools
	private:

	std::vector<std::vector<T>> kernel; //!< Kernel for convolution
	std::vector<int> dim_cube; //!< array containing the dimensions of the hypercube of spatial dimensions \f$ 2^n\_side \times 2^n\_side \f$ (for the multiresolution process) 
	std::vector<int> dim_data; //!< array containing the dimensions of the hypercube

	int dim_x;
	int dim_y;
	int dim_v;
	hypercube<T> file; //!< Dummy 

	double temps_global;
	double temps_modification_beta;
	double temps_f_g_cube;
	double temps_conv;
	double temps_deriv;
	double temps_tableaux;
	double temps_bfgs;
	double temps_update_beginning;
	double temps_;
	double temps_f_g_cube_tot;
	double temps_1_tot;
	double temps_2_tot;
	double temps_3_tot;
	double temps_tableau_update;
	double temps_setulb;
	double temps_transfert_d;
	double temps_copy;

	float temps_transfert;
	float temps_mirroirs;

	int n_gauss_add; 
	std::vector<T> std_spect, mean_spect, max_spect, max_spect_norm;
};

////START
template<typename T>
algo_rohsa<T>::algo_rohsa(parameters<T> &M, hypercube<T> &Hypercube)
{
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
	T max_mean_spect = *std::max_element(mean_spect.begin(), mean_spect.end());

	max_spectrum_norm(dim_data[0], dim_data[1], dim_data[2], max_mean_spect);

	// can't define the variable in the if  
	std::vector<std::vector<std::vector<T>>> grid_params, fit_params;

	if(M.descent){
		std::vector<std::vector<std::vector<T>>> grid_params_(3*(M.n_gauss+(file.nside*M.n_gauss_add)), std::vector<std::vector<T>>(dim_data[1], std::vector<T>(dim_data[0],0.)));
		std::vector<std::vector<std::vector<T>>> fit_params_(3*(M.n_gauss+(file.nside*M.n_gauss_add)), std::vector<std::vector<T>>(1, std::vector<T>(1,0.)));
		grid_params=grid_params_;
		fit_params=fit_params_;
	}
	else{
		std::vector<std::vector<std::vector<T>>> grid_params_(3*(M.n_gauss+M.n_gauss_add), std::vector<std::vector<T>>(dim_data[1], std::vector<T>(dim_data[0],0.)));
		grid_params=grid_params_;
	}

	if(M.descent)
	{
		std::cout<<"DÉBUT DE DESCENTE"<<std::endl;
		descente(M, grid_params, fit_params);
	}

	std::cout << "params.size() : " << fit_params.size() << " , " << fit_params[0].size() << " , " << fit_params[0][0].size() <<  std::endl;
	std::cout << "params.size() : " << grid_params.size() << " , " << grid_params[0].size() << " , " << grid_params[0][0].size() <<  std::endl;

	this->file.write_into_binary(M, this->grid_params);

}

//void algo_rohsa<T>::descente(parameters &M, std::vector<std::vector<std::vector<T>>> &grid_params, std::vector<std::vector<std::vector<T>>> &fit_params){	
template <typename T> 
void algo_rohsa<T>::descente(parameters<T> &M, std::vector<std::vector<std::vector<T>>> &grid_params, std::vector<std::vector<std::vector<T>>> &fit_params){	
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
	temps_copy = 0.; 
	for(int i=0;i<M.n_gauss; i++){
		fit_params[0+3*i][0][0] = 0.;
		fit_params[1+3*i][0][0] = 1.;
		fit_params[2+3*i][0][0] = 1.;
	}

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

	int n;
	double temps1_before_nside = omp_get_wtime();

		printf("M.lambda_amp = %f\n",M.lambda_amp);
		printf("M.lambda_mu = %f\n",M.lambda_mu);
		printf("M.lambda_sig = %f\n",M.lambda_sig);
		printf("M.lambda_var_amp = %f\n",M.lambda_var_amp);
		printf("M.lambda_var_mu = %f\n",M.lambda_var_mu);
		printf("M.lambda_var_sig = %f\n",M.lambda_var_sig);
		printf("M.amp_fact_init = %f\n",M.amp_fact_init);
		printf("M.sig_init = %f\n",M.sig_init);
		printf("M.n_gauss = %d\n",M.n_gauss);

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
				std::vector<double> cube_mean_flat_init_double(cube_mean[0][0].size());
				std::vector<double> fit_params_flat_init_double(fit_params.size(),0.); //used below
				for(int e(0); e<cube_mean[0][0].size(); e++) {
					cube_mean_flat_init_double[e] = double(cube_mean[0][0][e]); //cache ok
				}
				for(int e(0); e<fit_params_flat_init_double.size(); e++) {
					fit_params_flat_init_double[e] = double(fit_params[e][0][0]); //cache   USELESS SINCE NO ITERATION OCCURED BEFORE
				}


				//assume option "mean"
				std::cout<<"Init mean spectrum"<<std::endl;
				
				init_spectrum(M, cube_mean_flat_init_double, fit_params_flat_init_double);
	//				init_spectrum(M, cube_mean_flat, std_spect); //option spectre
	//				init_spectrum(M, cube_mean_flat, max_spect); //option max spectre
	//				init_spectrum(M, cube_mean_flat, max_spect_norm); //option norme spectre
/*
				for(int e(0); e<fit_params_flat_init_double.size(); e++) {
					printf("fit_params_flat_init_double[%d] = %.16f\n",e,fit_params_flat_init_double[e]); //cache   USELESS SINCE NO ITERATION OCCURED BEFORE
				}
				std::cin.ignore();
*/
//				exit(0);

				for(int i(0); i<M.n_gauss; i++) {
						b_params[i]= T(fit_params_flat_init_double[2+3*i]);
					}

				//we recover fit_params from its 1D version since we can't do fit_params[.][1][1] in C/C++
				for(int e(0); e<fit_params_flat_init_double.size(); e++) {
					fit_params[e][0][0] = T(fit_params_flat_init_double[e]); //cache
					}
				}
			
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

/*
				for(int e(0); e<fit_params.size(); e++) {
					printf("fit_params[%d][0][0] = %.16f\n",e,fit_params[e][0][0]); //cache   USELESS SINCE NO ITERATION OCCURED BEFORE
				}
				std::cin.ignore();
*/


						double temps2_upgrade = omp_get_wtime();
						temps_upgrade+=temps2_upgrade-temps1_upgrade;
/*
fit_params[0][0][0] = 2.0050162089404973;
fit_params[1][0][0] = 34.8209155748459693;
fit_params[2][0][0] = 11.4670363365702190;
fit_params[3][0][0] = 1.6013485980614408;
fit_params[4][0][0] = 103.6737248113834653;
fit_params[5][0][0] = 10.5077433796586632;
fit_params[6][0][0] = 1.7215169479274277;
fit_params[7][0][0] = 37.5691112042946287;
fit_params[8][0][0] = 1.6390353288355533;
fit_params[9][0][0] = 7.1863201795084928;
fit_params[10][0][0] = 34.5595485571460301;
fit_params[11][0][0] = 1.8297352300393890;
fit_params[12][0][0] = 2.7476664912880673;
fit_params[13][0][0] = 101.0331758355382021;
fit_params[14][0][0] = 3.5603237833754138;
fit_params[15][0][0] = 0.2752223659412244;
fit_params[16][0][0] = 68.4359932735529952;
fit_params[17][0][0] = 10.7975758827606487;
fit_params[18][0][0] = 1.2960773402025128;
fit_params[19][0][0] = 30.2952212742054812;
fit_params[20][0][0] = 1.2837584049698683;
fit_params[21][0][0] = 0.3547003275216601;
fit_params[22][0][0] = 108.5435787843823050;
fit_params[23][0][0] = 2.7916331327471609;
fit_params[24][0][0] = 14.1474872742121605;
fit_params[25][0][0] = 35.8097720768966497;
fit_params[26][0][0] = 4.0796185552538837;
fit_params[27][0][0] = 0.0507513970882115;
fit_params[28][0][0] = 1.4801271879387774;
fit_params[29][0][0] = 3.4379646715313328;
fit_params[30][0][0] = 0.4073933955858907;
fit_params[31][0][0] = 21.8262737831335834;
fit_params[32][0][0] = 2.9227478705517469;
fit_params[33][0][0] = 0.1039070910971688;
fit_params[34][0][0] = 47.0012069960593806;
fit_params[35][0][0] = 2.1412268676013992;
*/
					}
					if (n>0 and n<file.nside){
/*
			std::cout<<"cube_mean[0][0][0] = "<<cube_mean[0][0][0]<<std::endl;
			std::cout<<"cube_mean[1][0][0] = "<<cube_mean[1][0][0]<<std::endl;
			std::cout<<"cube_mean[0][1][0] = "<<cube_mean[0][1][0]<<std::endl;
			std::cout<<"cube_mean[0][0][1] = "<<cube_mean[0][0][1]<<std::endl;
*/

						if (power >=2){
/*			std::cout<<"fit_params[0][0][0] = "<<fit_params[0][0][0]<<std::endl;
			std::cout<<"fit_params[0][0][1] = "<<fit_params[0][0][1]<<std::endl;
			std::cout<<"fit_params[0][1][0] = "<<fit_params[0][1][0]<<std::endl;
			std::cout<<"fit_params[0][1][1] = "<<fit_params[0][1][1]<<std::endl;
			std::cout<<"fit_params[0][0][2] = "<<fit_params[0][0][2]<<std::endl;
			std::cout<<"fit_params[0][2][0] = "<<fit_params[0][2][0]<<std::endl;
			std::cout<<"fit_params[0][2][2] = "<<fit_params[0][2][2]<<std::endl;
			std::cout<<"fit_params[0][0][3] = "<<fit_params[0][0][3]<<std::endl;
			std::cout<<"fit_params[0][3][0] = "<<fit_params[0][3][0]<<std::endl;
			std::cout<<"fit_params[0][3][3] = "<<fit_params[0][3][3]<<std::endl;
*/
						}
						std::vector<std::vector<T>> std_map(power, std::vector<T>(power,0.));
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

						update_clean(M, cube_mean, fit_params, std_map, power, power, dim_v, b_params);

	//					(this->file).plot_multi_lines(fit_params, cube_mean, std::to_string(power));

						double temps2_update_pp=omp_get_wtime();
						temps_update_pp += temps2_update_pp-temps1_update_pp;

						if (M.n_gauss_add != 0){
							//Add new Gaussian if one reduced chi square > 1  
						}
						if(M.print_mean_parameters){	
							mean_parameters(fit_params);
						}

/*
		std::cout<<"fit_params[0][0][0] = "<<fit_params[0][0][0]<<std::endl;
		std::cout<<"fit_params[1][0][0] = "<<fit_params[1][0][0]<<std::endl;
		std::cout<<"fit_params[0][1][0] = "<<fit_params[0][1][0]<<std::endl;
		std::cout<<"fit_params[0][0][1] = "<<fit_params[0][0][1]<<std::endl;
		std::cin.ignore();
*/
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

		temps2_before_nside = omp_get_wtime();
		std::cout<<"                                "<<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"            Halfway through multiresolution             "<<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"Time spent in the multiresolution process up to this point : "<<omp_get_wtime() - temps1_descente <<std::endl;
		std::cout<<"	-> Computation time init_spectrum : "<< temps_init_spectrum <<std::endl;
		std::cout<<"	-> Computation time upgrade function (update 1D) : "<< temps_upgrade <<std::endl;
		std::cout<<"	-> Computation time std_map : "<< temps_std_map_pp <<std::endl;
		std::cout<<"	-> Computation time update (update 1->n-1) : "<< temps_update_pp <<std::endl;
		std::cout<<"	-> Computation time go_up_level (grid k->k+1) : "<<temps_go_up_level <<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"            Details             "<<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"Computation time setulb : "<<temps_setulb<<std::endl;
		std::cout<<"Computation time f_g_cube : "<<this->temps_f_g_cube<<std::endl;
		std::cout<< "	-> Computation time data transfer : " << this->temps_copy  <<std::endl;
		std::cout<< "	-> Computation time residual and residual term of the cost function : " << this->temps_tableaux <<std::endl;
		std::cout<< "	-> Computation time gradient (only data contribution to g) : " << this->temps_deriv  <<std::endl;
		std::cout<< "	-> Computation time regularization (spatial coherence contribution to f and g) : " << this->temps_conv <<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"           End details             "<<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"                                "<<std::endl;
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
		set_stdmap(std_map, this->file.data, M.lstd, M.ustd);
	}
	double temps_std_map2=omp_get_wtime();
	temps_std_map_dp+=temps_std_map2-temps_std_map1;

	double temps_update_dp1 = omp_get_wtime();

	if(print){
		int number_plot_2D = ceil(log(this->dim_data[1])/log(2));
		this->file.simple_plot_through_regu(grid_params, 0,0,number_plot_2D, "début");
	//	this->file.simple_plot_through_regu(grid_params, 0,1,number_plot_2D, "début");
	//	this->file.simple_plot_through_regu(grid_params, 0,2,number_plot_2D, "début");
		this->file.save_result_multires(grid_params, M, number_plot_2D);
	}

	if(M.regul){
		std::cout<<"Updating last level"<<std::endl;
		update_clean(M, this->file.data, grid_params, std_map, this->dim_data[0], this->dim_data[1], this->dim_v, b_params);
	}

	if(print){
		int number_plot_2D = ceil(log(this->dim_data[1])/log(2));
		this->file.simple_plot_through_regu(grid_params, 0,0,number_plot_2D, "fin");
	//	this->file.simple_plot_through_regu(grid_params, 0,1,number_plot_2D, "fin");
	//	this->file.simple_plot_through_regu(grid_params, 0,2,number_plot_2D, "fin");
		this->file.save_result_multires(grid_params, M, number_plot_2D);
	}

		double temps_update_dp2 = omp_get_wtime();
		temps_update_dp +=temps_update_dp2-temps_update_dp1;
	
		this->grid_params = grid_params;
		int comptage = 600;

		double temps2_descente = omp_get_wtime();
		double temps_dernier_niveau2 = omp_get_wtime();
		temps_dernier_niveau+=temps_dernier_niveau2-temps_dernier_niveau1;


		std::cout<<"                                "<<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"            End of multiresolution             "<<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"Total multiresolution computation time : "<<temps2_descente - temps1_descente <<std::endl;
		std::cout<<"	-> Computation time init_spectrum : "<< temps_init_spectrum <<std::endl;
		std::cout<<"	-> Computation time upgrade function (update 1D) : "<< temps_upgrade <<std::endl;
		std::cout<<"	-> Computation time std_map : "<< temps_std_map_pp <<std::endl;
		std::cout<<"	-> Computation time update (update 1->n-1) : "<< temps_update_pp <<std::endl;
		std::cout<<"	-> Computation time go_up_level (grid k->k+1) : "<<temps_go_up_level <<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"Computation time from levels 1 to n-1 : "<<temps2_before_nside - temps1_before_nside <<std::endl;
		std::cout<<"Time spent on reshape_down function (n-1 -> n)"<<temps_reshape_down <<std::endl;
		std::cout<<"Time spent on the last level n : "<< temps_dernier_niveau <<std::endl;
		std::cout<<"	-> Computation time std_map : "<< temps_std_map_dp <<std::endl;
		std::cout<<"	-> Computation time update : "<< temps_update_dp <<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"            Details             "<<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"Computation time setulb : "<<temps_setulb<<std::endl;
		std::cout<<"Computation time f_g_cube : "<<this->temps_f_g_cube<<std::endl;
		std::cout<< "	-> Computation time data transfer : " << this->temps_copy  <<std::endl;
		std::cout<< "	-> Computation time residual and residual term of the cost function : " << this->temps_tableaux <<std::endl;
		std::cout<< "	-> Computation time gradient (only data contribution to g) : " << this->temps_deriv  <<std::endl;
		std::cout<< "	-> Computation time regularization (spatial coherence contribution to f and g) : " << this->temps_conv <<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"           End of details section             "<<std::endl;
		std::cout<<"                                "<<std::endl;
		std::cout<<"                                "<<std::endl;
	}


template <typename T> 
void algo_rohsa<T>::reshape_down(std::vector<std::vector<std::vector<T>>> &tab1, std::vector<std::vector<std::vector<T>>>&tab2)
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


template <typename T> 
void algo_rohsa<T>::update_clean(parameters<T> &M, std::vector<std::vector<std::vector<T>>> &cube_avgd_or_data, std::vector<std::vector<std::vector<T>>> &params, std::vector<std::vector<T>> &std_map, int indice_x, int indice_y, int indice_v, std::vector<T> &b_params) {

	int n_beta = (3*M.n_gauss * indice_y * indice_x) +M.n_gauss;

	double temps1_tableau_update = omp_get_wtime();

	std::vector<T> cube_flat(cube_avgd_or_data[0][0].size(),0.);
	std::vector<std::vector<std::vector<T>>> lb_3D(3*M.n_gauss, std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<std::vector<std::vector<T>>> ub_3D(3*M.n_gauss, std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::vector<T> lb_3D_flat(lb_3D.size(),0.);
	std::vector<T> ub_3D_flat(ub_3D.size(),0.);
//	std::cout << "lb_3D.size() : " << lb_3D.size() << " , " << lb_3D[0].size() << " , " << lb_3D[0][0].size() <<  std::endl;
//	std::cout << "ub_3D.size() : " << ub_3D.size() << " , " << ub_3D[0].size() << " , " << ub_3D[0][0].size() <<  std::endl;

	if(M.wrapper){
		double* cube_flattened = NULL;
		size_t size_cube = indice_x*indice_y*indice_v*sizeof(double); 
		cube_flattened = (double*)malloc(size_cube);
		std::vector<std::vector<std::vector<double>>> cube_avgd_or_data_double(indice_x, std::vector<std::vector<double>>(indice_y, std::vector<double>(indice_v,0.)));
		for(int i=0; i<indice_x; i++) {
			for(int j=0; j<indice_y; j++) {
				for(int k=0; k<indice_v; k++) {
					cube_avgd_or_data_double[i][j][k] = double(cube_avgd_or_data[i][j][k]);
					cube_flattened[k*indice_x*indice_y+j*indice_x+i] = cube_avgd_or_data[i][j][k];
				}
			}
		}

		for(int j=0; j<indice_x; j++) {
			for(int i=0; i<indice_y; i++) {
				for(int p=0; p<cube_flat.size(); p++){
					cube_flat[p]=cube_avgd_or_data_double[j][i][p];
				}
				for(int p=0; p<3*M.n_gauss; p++){
					lb_3D_flat[p]=lb_3D[p][i][j];
					ub_3D_flat[p]=ub_3D[p][i][j];
				}
				init_bounds(M, cube_flat, M.n_gauss, lb_3D_flat, ub_3D_flat, false); //bool _init = false 
				for(int p=0; p<3*M.n_gauss; p++){
					lb_3D[p][i][j]=lb_3D_flat[p]; // the indices have been inverted
					ub_3D[p][i][j]=ub_3D_flat[p]; //
				}
			}
		}

		parameters<double> M_d;
		M_d.filename_dat = M.filename_dat;
		M_d.filename_fits = M.filename_fits;
		M_d.file_type_dat_check = M.file_type_dat_check;
		M_d.file_type_fits_check = M.file_type_fits_check;
		M_d.slice_index_min = M.slice_index_min;
		M_d.fileout = M.fileout;
		M_d.filename_noise = M.filename_noise;
		M_d.n_gauss = M.n_gauss;
		M_d.lambda_amp = double(M.lambda_amp);
		M_d.lambda_mu = double(M.lambda_mu);
		M_d.lambda_sig = double(M.lambda_sig);
		M_d.lambda_var_amp = double(M.lambda_var_amp);
		M_d.lambda_var_mu = double(M.lambda_var_mu);
		M_d.lambda_var_sig = double(M.lambda_var_sig);
		M_d.amp_fact_init = double(M.amp_fact_init);
		M_d.sig_init = double(M.sig_init);
		M_d.init_option = M.init_option;
		M_d.maxiter_init = M.maxiter_init;
		M_d.maxiter = M.maxiter;
		M_d.m = M.m;
		M_d.check_noise = M.check_noise;
		M_d.check_regul = M.check_regul;
		M_d.check_descent = M.check_descent;
		M_d.lstd = M.lstd;
		M_d.ustd = M.ustd;
		M_d.iprint = M.iprint;
		M_d.iprint_init = M.iprint_init;
		M_d.check_save_grid = M.check_save_grid;
		M_d.ub_sig = double(M.ub_sig);
		M_d.lb_sig = double(M.lb_sig);
		M_d.ub_sig_init = double(M.ub_sig_init);
		M_d.lb_sig_init = double(M.lb_sig_init);
		M_d.double_or_float = M.double_or_float;
		M_d.select_version = M.select_version;
		//		(this->M).copy_double_T(M_double, M);

		double* std_map_ = NULL;
		std_map_ = (double*)malloc(indice_x*indice_y*sizeof(double)); 
		for(int i=0; i<indice_y; i++){
			for(int j=0; j<indice_x; j++){
				std_map_[i*indice_x+j]=double(std_map[i][j]);
			}
		}

		double* lb = NULL;
		double* ub = NULL;
		double* beta = NULL;
		size_t size_n_beta = n_beta*sizeof(double); 
		lb = (double*)malloc(size_n_beta);
		ub = (double*)malloc(size_n_beta);
		beta = (double*)malloc(size_n_beta);
		initialize_array_double(lb, n_beta, 0.);
		initialize_array_double(ub, n_beta, 0.);
		initialize_array_double(beta, n_beta, 0.);		//three_D_to_one_D_inverted_dimensions<T>(threeD, oneD, x,y,z) transforms a 3D array of dimensions z,y,x into a 1D array of dimensions z,y,x
		if(M.select_version == 0){ //-cpu
			three_D_to_one_D_inverted_dimensions_double(lb_3D, lb, 3*M.n_gauss, indice_y, indice_x);
			three_D_to_one_D_inverted_dimensions_double(ub_3D, ub, 3*M.n_gauss, indice_y, indice_x);
			three_D_to_one_D_inverted_dimensions_double(params, beta, 3*M.n_gauss, indice_y, indice_x);
		}else if(M.select_version == 1 || M.select_version == 2){ //-gpu
			//three_D_to_one_D_same_dimensions(threeD, oneD, x,y,z) transforms a 3D array of dimensions z,y,x into a 1D array of dimensions x,y,z
			//lb 1difié est du format x, y, 3*ng
			three_D_to_one_D_same_dimensions_double(lb_3D, lb, 3*M.n_gauss, indice_y, indice_x);
			three_D_to_one_D_same_dimensions_double(ub_3D, ub, 3*M.n_gauss, indice_y, indice_x);
			three_D_to_one_D_same_dimensions_double(params, beta, 3*M.n_gauss, indice_y, indice_x);
		}
		for(int i=0; i<M.n_gauss; i++){
			lb[n_beta-M.n_gauss+i] = double(M.lb_sig);
			ub[n_beta-M.n_gauss+i] = double(M.ub_sig);
			beta[n_beta-M.n_gauss+i] = double(b_params[i]);
		}
		if(print){
			int number_plot_2D = ceil(log(indice_x)/log(2));
			this->file.simple_plot_through_regu(params, 0,0,number_plot_2D, "début");
//			this->file.simple_plot_through_regu(params, 0,1,number_plot_2D, "début");
//			this->file.simple_plot_through_regu(params, 0,2,number_plot_2D, "début");
			this->file.save_result_multires(params, M, number_plot_2D);
		}
		minimize_clean(M_d, n_beta, M.m, beta, lb, ub, cube_avgd_or_data_double, std_map_, indice_x, indice_y, indice_v, cube_flattened);
		if(M.select_version == 0){ //-cpu
			one_D_to_three_D_inverted_dimensions_double(beta, params, 3*M.n_gauss, indice_y, indice_x);
		}else if(M.select_version == 1 || M.select_version == 2){ //-gpu and -h
			one_D_to_three_D_same_dimensions_double(beta, params, 3*M.n_gauss, indice_y, indice_x);
		}
		if(print){
			int number_plot_2D = ceil(log(indice_x)/log(2));
			this->file.simple_plot_through_regu(params, 0,0,number_plot_2D, "fin");
//			this->file.simple_plot_through_regu(params, 0,1,number_plot_2D, "fin");
//			this->file.simple_plot_through_regu(params, 0,2,number_plot_2D, "fin");
			this->file.save_result_multires(params, M, number_plot_2D);
		}
		for(int i=0; i<M.n_gauss; i++){
			b_params[i]=T(beta[n_beta-M.n_gauss+i]);
		}
		free(lb);
		free(ub);
		free(beta);
		free(cube_flattened);
		free(std_map_);
	}else{
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
		for(int j=0; j<indice_x; j++) {
			for(int i=0; i<indice_y; i++) {
				for(int p=0; p<cube_flat.size(); p++){
					cube_flat[p]=cube_avgd_or_data[j][i][p];
				}
				for(int p=0; p<3*M.n_gauss; p++){
					lb_3D_flat[p]=lb_3D[p][i][j];
					ub_3D_flat[p]=ub_3D[p][i][j];
				}
				init_bounds(M, cube_flat, M.n_gauss, lb_3D_flat, ub_3D_flat, false); //bool _init = false 
				for(int p=0; p<3*M.n_gauss; p++){
					lb_3D[p][i][j]=lb_3D_flat[p]; // the indices have been inverted
					ub_3D[p][i][j]=ub_3D_flat[p]; //
				}
			}
		}
		T* lb = NULL;
		T* ub = NULL;
		T* beta = NULL;
		size_t size_n_beta = n_beta*sizeof(T); 
		lb = (T*)malloc(size_n_beta);
		ub = (T*)malloc(size_n_beta);
		beta = (T*)malloc(size_n_beta);
		initialize_array(lb, n_beta, 0.);
		initialize_array(ub, n_beta, 0.);
		initialize_array(beta, n_beta, 0.);		//three_D_to_one_D_inverted_dimensions<T>(threeD, oneD, x,y,z) transforms a 3D array of dimensions z,y,x into a 1D array of dimensions z,y,x
		if(M.select_version == 0){ //-cpu
			three_D_to_one_D_inverted_dimensions(lb_3D, lb, 3*M.n_gauss, indice_y, indice_x);
			three_D_to_one_D_inverted_dimensions(ub_3D, ub, 3*M.n_gauss, indice_y, indice_x);
			three_D_to_one_D_inverted_dimensions(params, beta, 3*M.n_gauss, indice_y, indice_x);
		}else if(M.select_version == 1 || M.select_version == 2){ //-gpu
			//three_D_to_one_D_same_dimensions(threeD, oneD, x,y,z) transforms a 3D array of dimensions z,y,x into a 1D array of dimensions x,y,z
			//lb 1difié est du format x, y, 3*ng
			three_D_to_one_D_same_dimensions(lb_3D, lb, 3*M.n_gauss, indice_y, indice_x);
			three_D_to_one_D_same_dimensions(ub_3D, ub, 3*M.n_gauss, indice_y, indice_x);
			three_D_to_one_D_same_dimensions(params, beta, 3*M.n_gauss, indice_y, indice_x);
		}
		for(int i=0; i<M.n_gauss; i++){
			lb[n_beta-M.n_gauss+i] = M.lb_sig;
			ub[n_beta-M.n_gauss+i] = M.ub_sig;
			beta[n_beta-M.n_gauss+i] = b_params[i];
		}
		if(print){
			int number_plot_2D = ceil(log(indice_x)/log(2));
			this->file.simple_plot_through_regu(params, 0,0,number_plot_2D, "début");
//			this->file.simple_plot_through_regu(params, 0,1,number_plot_2D, "début");
//			this->file.simple_plot_through_regu(params, 0,2,number_plot_2D, "début");
			this->file.save_result_multires(params, M, number_plot_2D);
		}
//		minimize_clean_gpu(M, n_beta, M.m, beta, lb, ub, cube_avgd_or_data, std_map, indice_x, indice_y, indice_v, cube_flattened); 
		minimize_clean_cpu(M, n_beta, M.m, beta, lb, ub, cube_avgd_or_data, std_map, indice_x, indice_y, indice_v, cube_flattened); 

//		void algo_rohsa<T>::minimize_clean_same_dim_test(parameters<double> &M, long n, long m, double* beta, double* lb, double* ub, std::vector<std::vector<std::vector<double>>> &cube, double* std_map_, int dim_x, int dim_y, int dim_v, double* cube_flattened) {

		if(M.select_version == 0){ //-cpu
			one_D_to_three_D_inverted_dimensions(beta, params, 3*M.n_gauss, indice_y, indice_x);
		}else if(M.select_version == 1 || M.select_version == 2){ //-gpu and -h
			one_D_to_three_D_same_dimensions(beta, params, 3*M.n_gauss, indice_y, indice_x);
		}
		if(print){
			int number_plot_2D = ceil(log(indice_x)/log(2));
			this->file.simple_plot_through_regu(params, 0,0,number_plot_2D, "fin");
//			this->file.simple_plot_through_regu(params, 0,1,number_plot_2D, "fin");
//			this->file.simple_plot_through_regu(params, 0,2,number_plot_2D, "fin");
			this->file.save_result_multires(params, M, number_plot_2D);
		}
		for(int i=0; i<M.n_gauss; i++){
			b_params[i]=beta[n_beta-M.n_gauss+i];
		}
		free(lb);
		free(ub);
		free(beta);
		free(cube_flattened);
	}
	temps_tableau_update += omp_get_wtime() - temps1_tableau_update;
}

template <typename T> 
void algo_rohsa<T>::set_stdmap(std::vector<std::vector<T>> &std_map, std::vector<std::vector<std::vector<T>>> &cube_or_data, int lb, int ub){
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
template <typename T> 
void algo_rohsa<T>::set_stdmap_transpose(std::vector<std::vector<T>> &std_map, std::vector<std::vector<std::vector<T>>> &cube, int lb, int ub){
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












template <typename T> 
void algo_rohsa<T>::minimize_clean(parameters<double> &M, long n, long m, double* beta, double* lb, double* ub, std::vector<std::vector<std::vector<double>>> &cube, double* std_map_, int dim_x, int dim_y, int dim_v, double* cube_flattened) {
    int i__1;
	int  i__c = 0;
    double d__1, d__2;

    double t1, t2, f;

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

	double* g = (double*)malloc(n*sizeof(double));
    for(int i(0); i<n; i++) {
	g[i]=0.;
    } 
    f=0.;

	temps_tableau_update += omp_get_wtime() - temps2_tableau_update;

    factr = 1e+7;
    pgtol = 1e-5;

    for (i__ = 0; i__ < n; i__ ++) {
        nbd[i__] = 2;
    }

    *task = (long)START;

	L111:

	double temps1_f_g_cube = omp_get_wtime();

	std::vector<std::vector<double>> std_map(dim_y, std::vector<double>(dim_x,0.));
	for(int i=0; i<dim_y; i++){
		for(int j=0; j<dim_x; j++){
			std_map[i][j] = std_map_[i*dim_x+j];
		}
	}

	double* std_map_dev = NULL;
	double* cube_flattened_dev = NULL;	
	checkCudaErrors(cudaMalloc(&std_map_dev, dim_x*dim_y*sizeof(double)));
	checkCudaErrors(cudaMemcpy(std_map_dev, std_map_, dim_x*dim_y*sizeof(double), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc(&cube_flattened_dev, dim_x*dim_y*dim_v*sizeof(double)));
	checkCudaErrors(cudaMemcpy(cube_flattened_dev, cube_flattened, dim_x*dim_y*dim_v*sizeof(double), cudaMemcpyHostToDevice));

	int compteur_iter_boucle_optim = 0;

	printf("dim_x = %d , dim_y = %d , dim_v = %d \n", dim_x, dim_y, dim_v);
	printf("n = %d , n_gauss = %d\n", int(n), M.n_gauss);

	if (true){//dim_x >128){
		printf("beta[n_beta-1] = %f , beta[n_beta] = %f\n", beta[n-1], beta[n-1]);
		printf("cube_flattened[dim_x*dim_y*dim_v-1] = %f , cube_flattened[dim_x*dim_y*dim_v] = %f\n", cube_flattened[dim_x*dim_y*dim_v-1], cube_flattened[dim_x*dim_y*dim_v]);
	}

    double* temps = NULL;
    temps = (double*)malloc(4*sizeof(double)); 
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

	if(false){//dim_x<64){
		f_g_cube_fast_unidimensional<double>(M, f, g, n, cube_flattened, cube, beta, dim_v, dim_y, dim_x, std_map_, temps);
//		f_g_cube_fast_clean(M, f, g, n, cube, beta, dim_v, dim_y, dim_x, std_map);
//		f_g_cube_cuda_L_clean(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened); // expérimentation gradient
	}else{
		if(M.select_version == 0){ //-cpu
			f_g_cube_fast_unidimensional<double>(M, f, g, n, cube_flattened, cube, beta, dim_v, dim_y, dim_x, std_map_, temps);
		}else if(M.select_version == 1){ //-gpu
			double* x_dev = nullptr;
			double* g_dev = nullptr;
			cudaMalloc(&x_dev, n * sizeof(x_dev[0]));
			cudaMalloc(&g_dev, n * sizeof(g_dev[0]));
			checkCudaErrors(cudaMemset(g_dev, 0., n*sizeof(g_dev[0])));
			checkCudaErrors(cudaMemcpy(x_dev, beta, n*sizeof(double), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaDeviceSynchronize());
			f_g_cube_parallel_lib<double>(M, f, g_dev, int(n), x_dev, int(dim_v), int(dim_y), int(dim_x), std_map_dev, cube_flattened_dev, temps);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaMemcpy(g, g_dev, n*sizeof(double), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaFree(x_dev));
			checkCudaErrors(cudaFree(g_dev));
			checkCudaErrors(cudaDeviceSynchronize());
		}else if(M.select_version == 2){ //-h
//			f_g_cube_not_very_fast_clean<T>(M, f, g, n, cube, beta, dim_v, dim_y, dim_x, std_map);
			f_g_cube_cuda_L_clean<double>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened, temps, this->temps_transfert_d, this->temps_mirroirs); // expérimentation gradient
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
			f_g_cube_fast_clean_optim_CPU(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map);
		}else{
			f_g_cube_cuda_L_clean(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened); 
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
	this->temps_f_g_cube += (temps[0]+temps[1]+temps[2]+temps[3])/1000;

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
	free(temps);

	checkCudaErrors(cudaFree(std_map_dev));
	checkCudaErrors(cudaFree(cube_flattened_dev));

	printf("----------------->temps_transfert_boucle = %f \n", temps_transfert_boucle);
	this->temps_transfert_d += temps_transfert_boucle;
}



template <typename T> 
void algo_rohsa<T>::minimize_clean_gpu(parameters<T> &M, long n, long m, T* beta, T* lb, T* ub, std::vector<std::vector<std::vector<T>>> &cube, std::vector<std::vector<T>> &std_map, int dim_x, int dim_y, int dim_v, T* cube_flattened) {

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

/*		int temp = 0;
		for(int ind = 0; ind < n; ind++){
			if(isnan(beta[ind]) && temp == 0){
				checkCudaErrors(cudaMemcpy(x_dev, beta, n*sizeof(T), cudaMemcpyHostToDevice));
				temp=1;
			}
		}
*/
	    checkCudaErrors(cudaDeviceSynchronize());
		f_g_cube_parallel_lib<T>(M, f, g_dev, n, x_dev, dim_v, dim_y, dim_x, std_map_dev, cube_flattened_dev, temps);
	    checkCudaErrors(cudaDeviceSynchronize());

//		checkCudaErrors(cudaMemcpy(beta, x_dev, n*sizeof(T), cudaMemcpyDeviceToHost));


	
/*
        printf("LIB --> fin-chemin : f = %.16f\n", f);
	    std::cin.ignore();
	    checkCudaErrors(cudaDeviceSynchronize());
	    checkCudaErrors(cudaDeviceSynchronize());
*/
		//	f_g_cube_fast_clean_optim_CPU_lib(M, f, g, n, cube, x, dim_v, dim_y, dim_x, std_map, &assist_buffer_cpu);
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
//	double f_init = std::numeric_limits<T>::max();
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
	this->temps_f_g_cube += (temps[0]+temps[1]+temps[2]+temps[3])/1000;

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
}



template <typename T> 
void algo_rohsa<T>::minimize_clean_cpu(parameters<T> &M, long n, long m, T* beta, T* lb, T* ub, std::vector<std::vector<std::vector<T>>> &cube, std::vector<std::vector<T>> &std_map, int dim_x, int dim_y, int dim_v, T* cube_flattened) {

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
	lbfgsb_options.eps_f = static_cast<T>(1e-10);
	lbfgsb_options.eps_g = static_cast<T>(1e-10);
	lbfgsb_options.eps_x = static_cast<T>(1e-10);
	lbfgsb_options.max_iteration = M.maxiter;
    lbfgsb_options.step_scaling = 1.0;
	lbfgsb_options.hessian_approximate_dimension = M.m;
  	lbfgsb_options.machine_epsilon = 1e-10;
  	lbfgsb_options.machine_maximum = std::numeric_limits<T>::max();

	// initialize LBFGSB state
	LBFGSB_CUDA_STATE<T> state;
	memset(&state, 0, sizeof(state));
	T* assist_buffer_cpu = nullptr;

  	T minimal_f = std::numeric_limits<T>::max();
  	state.m_funcgrad_callback = [&assist_buffer_cpu, &minimal_f, this, &M, &n, &cube, &cube_flattened, &cube_flattened_dev,
&std_map, &std_map_dev, &std_map_, dim_x, dim_y, dim_v, &temps](
                                  T* x, T& f, T* g,
                                  const cudaStream_t& stream,
                                  const LBFGSB_CUDA_SUMMARY<T>& summary) -> int {

	if(false){//dim_x<64){
//		f_g_cube_fast_clean_optim_CPU(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, temps);
//			f_g_cube_cuda_L_clean<double>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened, temps, this->temps_transfert_d, this->temps_mirroirs); // expérimentation gradient

		f_g_cube_fast_clean<T>(M, f, g, n, cube, x, dim_v, dim_y, dim_x, std_map, temps);
//		f_g_cube_cuda_L_clean(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened, temps, this->temps_transfert_d, this->temps_mirroirs); // expérimentation gradient
//		f_g_cube_not_very_fast_clean<T>(M, f, g, n, cube, x, dim_v, dim_y, dim_x, std_map, temps);
	}else{
		if(M.select_version == 0){ //-cpu
			f_g_cube_fast_unidimensional<T>(M, f, g, n, cube_flattened, cube, x, dim_v, dim_y, dim_x, std_map_, temps);
		}else if(M.select_version == 1){ //-gpu
			T* x_dev = nullptr;
			T* g_dev = nullptr;
			cudaMalloc(&x_dev, n * sizeof(x_dev[0]));
			cudaMalloc(&g_dev, n * sizeof(g_dev[0]));
			checkCudaErrors(cudaMemset(g_dev, 0., n*sizeof(g_dev[0])));
			checkCudaErrors(cudaMemcpy(x_dev, x, n*sizeof(T), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaDeviceSynchronize());
//			f_g_cube_parallel_lib<T>(M, f, g_dev, int(n), x_dev, int(dim_v), int(dim_y), int(dim_x), std_map_dev, cube_flattened_dev, temps);
			checkCudaErrors(cudaDeviceSynchronize());
			checkCudaErrors(cudaMemcpy(g, g_dev, n*sizeof(T), cudaMemcpyDeviceToHost));
			checkCudaErrors(cudaFree(x_dev));
			checkCudaErrors(cudaFree(g_dev));
			checkCudaErrors(cudaDeviceSynchronize());
		}else if(M.select_version == 2){ //-autre
			f_g_cube_cuda_L_clean<T>(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, cube_flattened, temps, this->temps_transfert, this->temps_mirroirs); // expérimentation gradient
		}
	}
		if (summary.num_iteration % 100 == 0) {
		std::cout << "CPU iteration " << summary.num_iteration << " F: " << f
					<< std::endl;    
		}

		minimal_f = fmin(minimal_f, f);
//        minimal_f = f;

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
	//	f_g_cube_fast_clean_optim_CPU(M, f_init, nullptr, n, cube, beta, dim_v, dim_y, dim_x, std_map);
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
	this->temps_f_g_cube += (temps[0]+temps[1]+temps[2]+temps[3])/1000;

	printf("this->temps_copy = %f\n", this->temps_copy);
	printf("this->temps_tableaux = %f\n", this->temps_tableaux);
	printf("this->temps_deriv = %f\n", this->temps_deriv);
	printf("this->temps_conv = %f\n", this->temps_conv);
	printf("this->temps_f_g_cube = %f\n", this->temps_f_g_cube);

}



template <typename T> 
void algo_rohsa<T>::go_up_level(std::vector<std::vector<std::vector<T>>> &fit_params) {
		//dimensions of fit_params
	int dim[3];
	dim[2]=fit_params[0][0].size();
	dim[1]=fit_params[0].size();
	dim[0]=fit_params.size();
 
	std::vector<std::vector<std::vector<T>>> cube_params_down(dim[0],std::vector<std::vector<T>>(dim[1], std::vector<T>(dim[2],0.)));	

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
//	std::cout<<"Dimensions of std_map : "<<std_map.size()<<" , "<<std_map[0].size()<<std::endl;

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

template <typename T> 
void algo_rohsa<T>::upgrade(parameters<T> &M, std::vector<std::vector<std::vector<T>>> &cube, std::vector<std::vector<std::vector<T>>> &params, int power) {
	int i,j;
    std::vector<double> line(dim_v,0.);
    std::vector<double> x(3*M.n_gauss,0.);
	
    std::vector<double> lb(3*M.n_gauss,0.);
    std::vector<double> ub(3*M.n_gauss,0.);

    for(i=0;i<power; i++){ //dim_x
    	for(j=0;j<power; j++){ //dim_y
        	int p;
            for(p=0; p<cube[0][0].size();p++){
            	line[p]=double(cube[i][j][p]); //???£
            }
            for(p=0; p<3*M.n_gauss; p++){
            	x[p]=double(params[p][i][j]); //cache
            }
/*            for(p=0; p<params.size();p++){
				printf("x[%d] = %.16f\n", p, x[p]);
			}
			printf("--------------------------------\n");
*/
            init_bounds_double(M, line, M.n_gauss, lb, ub, false); //bool _init = false;
            minimize_spec(M,3*M.n_gauss ,M.m ,x ,lb , M.n_gauss, ub ,line);
/*            for(p=0; p<params.size();p++){
				printf("x[%d] = %.16f\n", p, x[p]);
            }
*/
//			std::cin.ignore();			

            for(p=0; p<3*M.n_gauss;p++){
            	params[p][i][j]=T(x[p]); //cache
            }
        }
    }
}

template <typename T> 
void algo_rohsa<T>::init_bounds(parameters<T> &M, std::vector<T> &line, int n_gauss_local, std::vector<T> &lb, std::vector<T> &ub, bool _init) {

	T max_line = *std::max_element(line.begin(), line.end());
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

template <typename T> 
void algo_rohsa<T>::init_bounds_double(parameters<T> &M, std::vector<double> &line, int n_gauss_local, std::vector<double> &lb, std::vector<double> &ub, bool _init) {

	double max_line = *std::max_element(line.begin(), line.end());
//	std::cout<<"max_line = "<<max_line<<std::endl;
//	std::cout<<"dim_v = "<<dim_v<<std::endl;
	for(int i(0); i<n_gauss_local; i++) {
		lb[0+3*i]=0.;
		ub[0+3*i]=max_line;

		lb[1+3*i]=0.;
		ub[1+3*i]=dim_v;
		if (_init){
			lb[2+3*i]=double(M.lb_sig_init);
			ub[2+3*i]=double(M.ub_sig_init);
		}else{
			lb[2+3*i]=double(M.lb_sig);
			ub[2+3*i]=double(M.ub_sig);
		}
	}
}

template <typename T> 
T algo_rohsa<T>::model_function(int x, T a, T m, T s) {
	return a*exp(-pow(T(x)-m,2.) / (2.*pow(s,2.)));
}

template <typename T> 
int algo_rohsa<T>::minloc(std::vector<T> &tab) {
	return std::distance(tab.begin(), std::min_element( tab.begin(), tab.end() ));
}

template <typename T> 
int algo_rohsa<T>::minloc_double(std::vector<double> &tab) {
	return std::distance(tab.begin(), std::min_element( tab.begin(), tab.end() ));
}

template <typename T> 
void algo_rohsa<T>::minimize_spec(parameters<T> &M, long n, long m, std::vector<double> &x_v, std::vector<double> &lb_v, int n_gauss_i, std::vector<double> &ub_v, std::vector<double> &line_v) {
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
    double x[x_v.size()];
    double lb[lb_v.size()];
    double ub[ub_v.size()];
    double line[line_v.size()];

    for(int i(0); i<line_v.size(); i++) {
	line[i]=line_v[i];
    } 
    for(int i(0); i<n; i++) {
	g[i]=0.;
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
			myresidual_double(x, line, _residual_, n_gauss_i);
			f = myfunc_spec_double(_residual_);
			mygrad_spec_double(g, _residual_, x, n_gauss_i);
/*
            for(int p=0; p<n;p++){
				printf("g[%d] = %.16f\n", p, g[p]);
			}
			printf("f = %.16f\n", f);
			std::cin.ignore();
*/
//			printf("f = %f\n", f);
//			printf("g[0] = %f\n", g[0]);
//			printf("g[1] = %f\n", g[1]);
//			printf("g[2] = %f\n", g[2]);
//			printf("g[3] = %f\n", g[3]);
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

template <typename T> 
void algo_rohsa<T>::minimize_spec_save(parameters<T> &M, long n, long m, std::vector<T> &x_v, std::vector<T> &lb_v, int n_gauss_i, std::vector<T> &ub_v, std::vector<T> &line_v) {
/* Minimize_spec */ 
//int MAIN__(void)
    std::vector<T> _residual_(dim_v,0.);
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
			myresidual_double(x, line, _residual_, n_gauss_i);
			f = myfunc_spec_double(_residual_);
			mygrad_spec_double(g, _residual_, x, n_gauss_i);
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

template <typename T> 
T algo_rohsa<T>::myfunc_spec(std::vector<T> &residual) {
	T S(0.);
	for(int p(0); p<residual.size(); p++) {
		S+=pow(residual[p],2);
	}
	return 0.5*S;
}


template <typename T> 
double algo_rohsa<T>::myfunc_spec_double(std::vector<double> &residual) {
	double S(0.);
	for(int p(0); p<residual.size(); p++) {
		S+=pow(residual[p],2);
	}
	return 0.5*S;
}


template <typename T> 
void algo_rohsa<T>::myresidual(T* params, T* line, std::vector<T> &residual, int n_gauss_i) {
	int i,k;
	std::vector<T> model(residual.size(),0.);
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

template <typename T> 
void algo_rohsa<T>::myresidual_double(double* params, double* line, std::vector<double> &residual, int n_gauss_i) {
	int k;
	std::vector<double> model(residual.size(),0.);
	for(int i(0); i<n_gauss_i; i++) {
		for(k=0; k<residual.size(); k++) {
			int nu = k+1;
			model[k]+= model_function(nu, params[3*i], params[1+3*i], params[2+3*i]);
		}
	}
	for(int p=0; p<residual.size(); p++) {
		residual[p]=model[p]-line[p]; 
	}
}

template <typename T> 
void algo_rohsa<T>::myresidual(std::vector<T> &params, std::vector<T> &line, std::vector<T> &residual, int n_gauss_i) {
	int k;
	std::vector<T> model(residual.size(),0.);
	for(int i(0); i<n_gauss_i; i++) {
		for(k=0; k<residual.size(); k++) {
			int nu = k+1;
			model[k]+= model_function(nu, params[3*i], params[1+3*i], params[2+3*i]);
		}
	}
	for(int p=0; p<residual.size(); p++) {
		residual[p]=model[p]-line[p]; 
	}
}

template <typename T> 
void algo_rohsa<T>::mygrad_spec(T gradient[], std::vector<T> &residual, T params[], int n_gauss_i) {
	std::vector<std::vector<T>> dF_over_dB(3*n_gauss_i, std::vector<T>(dim_v,0.));
	T g(0.);
	int i,k;
	for(int p(0); p<3*n_gauss_i; p++) {
		gradient[p]=0.;
	}
	for(i=0; i<n_gauss_i; i++) {
		for(int k(0); k<dim_v; k++) {
			T nu = T(k+1);
			dF_over_dB[0+3*i][k] += exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

			dF_over_dB[1+3*i][k] +=  params[3*i]*( nu - params[1+3*i])/pow(params[2+3*i],2.) * exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

			dF_over_dB[2+3*i][k] += params[3*i]*pow( nu - params[1+3*i] , 2.)/(pow(params[2+3*i],3.)) * exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );
		}
	}
	for(k=0; k<3*n_gauss_i; k++){
		for(int i=0; i<dim_v; i++){
			gradient[k]+=dF_over_dB[k][i]*residual[i];
		}
	}
}

template <typename T> 
void algo_rohsa<T>::mygrad_spec_double(double gradient[], std::vector<double> &residual, double params[], int n_gauss_i) {
	std::vector<std::vector<double>> dF_over_dB(3*n_gauss_i, std::vector<double>(dim_v,0.));
	double g(0.);
	int i,k;
	for(int p(0); p<3*n_gauss_i; p++) {
		gradient[p]=0.;
	}
	for(i=0; i<n_gauss_i; i++) {
		for(int k(0); k<dim_v; k++) {
			double nu = double(k+1);
			dF_over_dB[0+3*i][k] += exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

			dF_over_dB[1+3*i][k] +=  params[3*i]*( nu - params[1+3*i])/pow(params[2+3*i],2.) * exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

			dF_over_dB[2+3*i][k] += params[3*i]*pow( nu - params[1+3*i] , 2.)/(pow(params[2+3*i],3.)) * exp(-pow( nu-params[1+3*i],2.)/(2*pow(params[2+3*i],2.)) );

		}
	}
	for(k=0; k<3*n_gauss_i; k++){
		for(int i=0; i<dim_v; i++){
			gradient[k]+=dF_over_dB[k][i]*residual[i];
		}
	}
}

template <typename T> 
void algo_rohsa<T>::init_spectrum(parameters<T> &M, std::vector<double> &line, std::vector<double> &params) {	
	for(int i=1; i<=M.n_gauss; i++) {
		std::vector<double> model_tab(dim_v,0.);
		std::vector<double> residual(dim_v,0.);
		std::vector<double> lb(3*i,0.);
		std::vector<double> ub(3*i,0.);

		init_bounds_double(M, line,i,lb,ub, true); //we consider {bool _init = true;} since we want to initialize the boundaries

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
		
		double argmin_res = minloc_double(residual);
		x[0+3*(i-1)] = double(line[int(argmin_res)])*double(M.amp_fact_init);
		x[1+3*(i-1)] = double(argmin_res+1);
		x[2+3*(i-1)] = double(M.sig_init);
//		printf("--------------------------------\n");
//		for(int p(0); p<3*(i-1); p++){	
//			printf("x[%d] = %.16f\n",p,x[p]);
//		}
//		printf("--------------------------------\n");

		minimize_spec(M, 3*i, M.m, x, lb, i, ub, line);

//		for(int p(0); p<3*(i-1); p++){	
//			printf("x[%d] = %.16f\n",p,x[p]);
//		}

		for(int p(0); p<3*(i); p++){	
			params[p]=x[p];
		}
	}

}




template <typename T> 
void algo_rohsa<T>::mean_array(int power, std::vector<std::vector<std::vector<T>>> &cube_mean)
{
	std::vector<T> spectrum(file.dim_cube[2],0.);
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
						spectrum[m] += file.cube[l+j*n][k+i*n][m];
					}
				}
			}
			for(int m(0); m<file.dim_cube[2]; m++)
			{
				cube_mean[j][i][m] = spectrum[m]/pow(n,2);
			}
			for(int p(0); p<file.dim_cube[2]; p++)
			{
				spectrum[p] = 0.;
			}
		}
	}
}


//template <typename T> void algo_rohsa<T>::descente(parameters<T> &M, std::vector<std::vector<std::vector<T>>> &grid_params, std::vector<std::vector<std::vector<T>>> &fit_params){	

// // L'ordre x,y,lambda est celui du code fortran : lambda,y,x      pk?

// It transforms a 1D vector into a contiguous flattened 1D array from a 2D array, like a valarray
template <typename T> 
void algo_rohsa<T>::ravel_2D(const std::vector<std::vector<T>> &map, std::vector<T> &vector, int dim_y, int dim_x)
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


template <typename T> 
void algo_rohsa<T>::ravel_3D(const std::vector<std::vector<std::vector<T>>> &cube_3D, std::vector<T> &vector, int dim_v, int dim_y, int dim_x)
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

template <typename T>
void algo_rohsa<T>::initialize_array(T* array, int size, T value)
{
    for(int k=0; k<size; k++)
	{
		array[k]=value;
    }
}

template <typename T>
void algo_rohsa<T>::initialize_array_double(double* array, int size, double value)
{
    for(int k=0; k<size; k++)
	{
		array[k]=value;
    }
}

template <typename T> 
void algo_rohsa<T>::three_D_to_one_D(const std::vector<std::vector<std::vector<T>>> &cube_3D, std::vector<T> &vector, int dim_x, int dim_y, int dim_v)
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


template <typename T> 
void algo_rohsa<T>::three_D_to_one_D(const std::vector<std::vector<std::vector<T>>> &cube_3D, T* vector, int dim_x, int dim_y, int dim_v)
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


template <typename T> 
void algo_rohsa<T>::one_D_to_three_D_same_dimensions(T* vector, std::vector<std::vector<std::vector<T>>> &cube_3D, int dim_x, int dim_y, int dim_v)
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
//		one_D_to_three_D_inverted_dimensions<T>(beta, params, 3*M.n_gauss, indice_y, indice_x);

template <typename T> 
void algo_rohsa<T>::one_D_to_three_D_inverted_dimensions(T* vector, std::vector<std::vector<std::vector<T>>> &cube_3D, int dim_x, int dim_y, int dim_v)
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
void algo_rohsa<T>::one_D_to_three_D_same_dimensions_double(double* vector, std::vector<std::vector<std::vector<T>>> &cube_3D, int dim_x, int dim_y, int dim_v)
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
	            cube_3D[k][j][i] = T(vector[k*dim_y*dim_v+j*dim_v+i]);
				}
			}
	    }
	}
}
//		one_D_to_three_D_inverted_dimensions<T>(beta, params, 3*M.n_gauss, indice_y, indice_x);

template <typename T> 
void algo_rohsa<T>::one_D_to_three_D_inverted_dimensions_double(double* vector, std::vector<std::vector<std::vector<T>>> &cube_3D, int dim_x, int dim_y, int dim_v)
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
	            cube_3D[k][j][i] = T(vector[i*dim_y*dim_x+j*dim_x+k]);
				}
			}
	    }
	}
}

template <typename T> 
void algo_rohsa<T>::three_D_to_one_D_same_dimensions(const std::vector<std::vector<std::vector<T>>> &cube_3D, T* vector, int dim_x, int dim_y, int dim_v)
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
void algo_rohsa<T>::three_D_to_one_D_same_dimensions_double(const std::vector<std::vector<std::vector<T>>> &cube_3D, double* vector, int dim_x, int dim_y, int dim_v)
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
	            vector[k*dim_y*dim_v+j*dim_v+i] = double(cube_3D[k][j][i]);
				}
			}
	    }
	}
}

//		three_D_to_one_D_inverted_dimensions<T>(lb_3D, lb, 3*M.n_gauss, indice_y, indice_x);
template <typename T> 
void algo_rohsa<T>::three_D_to_one_D_inverted_dimensions(const std::vector<std::vector<std::vector<T>>> &cube_3D, T* vector, int dim_x, int dim_y, int dim_v)
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
	            vector[i*dim_x*dim_y+j*dim_x+k] = cube_3D[k][j][i];
				}
			}
	    }
	}
}

template <typename T> 
void algo_rohsa<T>::three_D_to_one_D_inverted_dimensions_double(const std::vector<std::vector<std::vector<T>>> &cube_3D, double* vector, int dim_x, int dim_y, int dim_v)
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
	            vector[i*dim_x*dim_y+j*dim_x+k] = double(cube_3D[k][j][i]);
				}
			}
	    }
	}
}

template <typename T> 
void algo_rohsa<T>::ravel_3D(const std::vector<std::vector<std::vector<T>>> &cube, T* vector, int dim_v, int dim_y, int dim_x)
{
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


// It transforms a 1D vector into a 3D array, like the step we went through when analysing data from CCfits which returns a valarray that needs to be expended into a 3D array (it's the data cube)
template <typename T> 
void algo_rohsa<T>::unravel_3D(const std::vector<T> &vector, std::vector<std::vector<std::vector<T>>> &cube, int dim_v, int dim_y, int dim_x)
{
	int k,j,i;
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


template <typename T> 
void algo_rohsa<T>::unravel_3D(T* vector, std::vector<std::vector<std::vector<T>>> &cube, int dim_v, int dim_y, int dim_x)
{
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


template <typename T> 
void algo_rohsa<T>::unravel_3D_T(T* vector, std::vector<std::vector<std::vector<T>>> &cube, int dim_x, int dim_y, int dim_z)
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
}

// It returns the mean value of a 1D vector
template <typename T> 
T algo_rohsa<T>::mean(const std::vector<T> &array)
{
 	return std::accumulate(array.begin(), array.end(), 0.)/std::max(T(1.),T(array.size()));
}

// It returns the standard deviation value of a 1D vector
// BEWARE THE STD LIBRARY 
// "Std" rather than "std"

template <typename T> 
T algo_rohsa<T>::Std(const std::vector<T> &array)
{
	T mean_(0.), var(0.);
	int n = array.size();
	mean_ = mean(array);

	for(int i(0); i<n; i++)
	{
		var+=pow(array[i]-mean_,2);
	}
	return sqrt(var/(n-1));
}

template <typename T> 
T algo_rohsa<T>::std_2D(const std::vector<std::vector<T>> &map, int dim_y, int dim_x)
{

	std::vector<T> vector(dim_x*dim_y, 0.);
	ravel_2D(map, vector, dim_y, dim_x);
	return Std(vector);
}

template <typename T> 
T algo_rohsa<T>::max_2D(const std::vector<std::vector<T>> &map, int dim_y, int dim_x)
{
	std::vector<T> vector(dim_x*dim_y,0.);
	ravel_2D(map, vector, dim_y, dim_x);
	T val_max = vector[0];
	for (unsigned int i = 0; i < vector.size(); i++)
		if (vector[i] > val_max)
    			val_max = vector[i];
	vector.clear();
	return val_max;
}

template <typename T> 
T algo_rohsa<T>::mean_2D(const std::vector<std::vector<T>> &map, int dim_y, int dim_x)
{
	std::vector<T> vector(dim_y*dim_x, 0.);
	ravel_2D(map, vector, dim_y, dim_x);
	T mean_2D = mean(vector);
	vector.clear();
	return mean_2D;
}


template <typename T> 
void algo_rohsa<T>::std_spectrum(int dim_x, int dim_y, int dim_v)
{
	for(int i(0); i<dim_v; i++)
	{
		std::vector<std::vector<T>> map(dim_x, std::vector<T>(dim_y,0.));
		for(int j(0); j< dim_y; j++)
		{
			for( int k(0); k<dim_x; k++)
			{
				map[k][j] = file.data[k][j][i];
			}
		}
		std_spect.vector<T>::push_back(std_2D(map, dim_y, dim_x));
	}
}


template <typename T> 
void algo_rohsa<T>::mean_spectrum(int dim_x, int dim_y, int dim_v)
{
	for(int i(0);i<dim_v;i++)
	{
		std::vector<std::vector<T>> map(dim_x, std::vector<T>(dim_y, 0.));
		for(int j(0); j<dim_y ; j++)
		{
			for(int k(0); k<dim_x ; k++)
			{
				map[k][j] = file.data[k][j][i];
			}
		}
		mean_spect.vector<T>::push_back(mean_2D(map, dim_y, dim_x));
		map.clear();
	}
}


template <typename T> 
void algo_rohsa<T>::max_spectrum(int dim_x, int dim_y, int dim_v)
{
	for(int i(0); i<dim_v; i++)
	{
		std::vector<std::vector<T>> map(dim_x, std::vector<T>(dim_y,0.));
		for(int j(0); j< dim_y; j++)
		{
			for( int k(0); k<dim_x; k++)
			{
				map[k][j] = file.data[k][j][i];
			}
		}
		max_spect.vector<T>::push_back(max_2D(map, dim_y, dim_x));
		map.clear();
	}
}

template <typename T> 
void algo_rohsa<T>::max_spectrum_norm(int dim_x, int dim_y, int dim_v, T norm_value)
{
	for(int i(0); i<dim_v; i++)
	{
		std::vector<std::vector<T>> map(dim_x, std::vector<T>(dim_y));
		for(int j(0); j< dim_y; j++)
		{
			for( int k(0); k<dim_x; k++)
			{
				map[k][j] = file.data[k][j][i];
			}
		}
		max_spect_norm.vector<T>::push_back(max_2D(map, dim_y, dim_x));
		map.clear();
	}

	T val_max = max_spect_norm[0];
	for (unsigned int i = 0; i < max_spect_norm.size(); i++)
		if (max_spect_norm[i] > val_max)
    			val_max = max_spect_norm[i];

	for(int i(0); i<dim_v ; i++)
	{
		max_spect_norm[i] /= val_max/norm_value; 
	}
}

template <typename T> 
void algo_rohsa<T>::mean_parameters(std::vector<std::vector<std::vector<T>>> &params)
{
	int dim1 = params.size();
	int dim2 = params[0].size();
	int dim3 = params[0][0].size();
	
	for(int p=0; p<dim1;p++){
		T mean = 0.;
		for(int i=0;i<dim2;i++){
			for(int j=0;j<dim3;j++){
				mean += params[p][i][j];
			}
		}
		mean = mean/(dim2*dim3);
		if (p%3 ==0)
			printf("Gaussian n°%d, parameter n°%d, mean a     = %f \n", (p-p%3)/3, p, mean);
		if (p%3 ==1)
			printf("Gaussian n°%d, parameter n°%d, mean mu    = %f \n", (p-p%3)/3, p, mean);
		if (p%3 ==2)
			printf("Gaussian n°%d, parameter n°%d, mean sigma = %f \n", (p-p%3)/3, p, mean);
	}

}



#endif
