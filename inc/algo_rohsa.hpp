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
#include <limits>
#include <lbfgsb_cpp/lbfgsb.hpp>
#include <array>

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

#define Nb_time_mes 10

//extern "C" void setulb_(int *n, int *m, double* x, double* l, double* u, int* nbd, double* f, double* g, double* factr, double* pgtol, double* wa, int* iwa, char *task, int* iprint, char *csave, bool *lsave, int *isave, double *dsave);

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
	algo_rohsa(std::vector<std::vector<T>> &std_map, parameters<T> &M, hypercube<T> &Hypercube);

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

	void descente(hypercube<T> &Hypercube, parameters<T> &M, std::vector<std::vector<std::vector<T>>> &grid_params, std::vector<std::vector<std::vector<T>>> &fit_params); //!< main loop for the multiresolution process
	void test_toolbox(hypercube<T> &Hypercube, parameters<T> &M, std::vector<std::vector<std::vector<T>>> &grid_params);

//	void descente(parameters<T> &M, std::vector<std::vector<std::vector<T>>> &grid_params, std::vector<std::vector<std::vector<T>>> &fit_params); //!< main loop for the multiresolution process
/**
 * @brief Similar to descente() but without regularization.
 *
 */
	void ravel_2D(const std::vector<std::vector<T>> &map, std::vector<T> &vector, int dim_y, int dim_x);
	void ravel_3D(const std::vector<std::vector<std::vector<T>>> &cube_3D, std::vector<T> &vector, int dim_v, int dim_y, int dim_x);

	void init_bounds(parameters<T> &M, std::vector<T>& line, int n_gauss_local, std::vector<T> &lb, std::vector<T> &ub, bool _init);
	void init_bounds_double(parameters<T> &M, std::vector<double>& line, int n_gauss_local, std::vector<double>& lb, std::vector<double>& ub, bool _init);

	void mean_array(hypercube<T> &Hypercube, int power, std::vector<std::vector<std::vector<T>>> &cube_mean);
	void reshape_noise_up(std::vector<std::vector<T>>& std_cube);
	void mean_noise_map(int n_side, int n, int power, std::vector<std::vector<T>> &std_cube, std::vector<std::vector<T>> &std_map);
	void init_spectrum(parameters<T> &M, std::vector<double> &line, std::vector<double> &params);
//	void init_spectrum(parameters<T> &M, std::vector<T> &line, std::vector<T> &params); //!< Initializes spectrum (called during first iteration)

	T model_function(int x, T a, T m, T s);

	int minloc(std::vector<T> &tab);
	int minloc_double(std::vector<double> &tab);
	void minimize_spec(parameters<T> &M, long n, long m, std::vector<double> &x_v, std::vector<double> &lb_v, int n_gauss_i, std::vector<double> &ub_v, std::vector<double> &line_v, int maxiter);
	//void minimize_spec(parameters<T> &M, long n, long m, std::vector<T> &x_v, std::vector<T> &lb_v, int n_gauss_i,std::vector<T> &ub_v, std::vector<T> &line_v); //!< Solves the optimization problem during the first iteration, it calls the L-BFGS-B black box.
	void minimize_spec_save(parameters<T> &M, long n, long m, std::vector<T> &x_v, std::vector<T> &lb_v, int n_gauss_i,std::vector<T> &ub_v, std::vector<T> &line_v, int maxiter); //!< Solves the optimization problem during the first iteration, it calls the L-BFGS-B black box.

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
	void update_clean(hypercube<T> Hypercube, parameters<T> &M, std::vector<std::vector<std::vector<T>>> &cube_avgd_or_data, std::vector<std::vector<std::vector<T>>> &params, std::vector<std::vector<T>> &std_map, int indice_x, int indice_y, int indice_v,std::vector<T> &b_params);//!< Prepares boundary conditions and calls the minimize function.

	void minimize_clean(parameters<double> &M, long n, long m, double* beta, double* lb, double* ub, std::vector<std::vector<std::vector<double>>> &cube, double* std_map_, int dim_x, int dim_y, int dim_v, double* cube_flattened);
	void minimize_clean_driver(parameters<double> &M, long n, long m, double* beta, double* lb, double* ub, std::vector<std::vector<std::vector<double>>> &cube, double* std_map_, int dim_x, int dim_y, int dim_v, double* cube_flattened);
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
	void std_spectrum(hypercube<T> &Hypercube, int dim_x, int dim_y, int dim_v);
	void mean_spectrum(hypercube<T> &Hypercube, int dim_x, int dim_y, int dim_v);
	void max_spectrum(hypercube<T> &Hypercube, int dim_x, int dim_y, int dim_v);
	void max_spectrum_norm(hypercube<T> &Hypercube, int dim_x, int dim_y, int dim_v, T norm_value);
	void mean_parameters(std::vector<std::vector<std::vector<T>>> &params);

	std::vector<std::vector<std::vector<T>>> grid_params; //!< 3D array containing the gaussian parameters \f$\lambda, \mu, \sigma \f$ depending on the spatial position. Dimensions : It is a \f$ 3 n\_gauss \times dim\_y \times dim\_x \f$.
	std::vector<std::vector<std::vector<T>>> fit_params; //!< same as grid_params (gaussian parameters) but this array is used through multiresolution. Dimensions : \f$ 3 n\_gauss \times 2^k \times 2^k \f$ for \f$ 0 < k < n\_side \f$.
	std::vector<std::vector<T>> std_data;
	std::vector<std::vector<std::vector<T>>> std_data_3D;
	std::vector<std::vector<T>> std_cube;
	std::vector<std::vector<std::vector<T>>> std_cube_3D;

	T* cube_or_dat_flattened; //!< Data flattened at each iteration. Because of the multiresolution process, it has to be computed at each iteration.

//	Computationnal tools
	private:

	std::vector<std::vector<T>> kernel; //!< Kernel for convolution
	std::vector<int> dim_cube; //!< array containing the dimensions of the hypercube of spatial dimensions \f$ 2^n\_side \times 2^n\_side \f$ (for the multiresolution process)
	std::vector<int> dim_data; //!< array containing the dimensions of the hypercube

	int dim_x;
	int dim_y;
	int dim_v;
////	hypercube<T> file; //!< Dummy

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
	double* temps_detail_regu;

	float temps_transfert;
	float temps_mirroirs;

	int n_gauss_add;
	std::vector<T> std_spect, mean_spect, max_spect, max_spect_norm;
};
template<typename T>
algo_rohsa<T>::algo_rohsa(std::vector<std::vector<T>> &std_map, parameters<T> &M, hypercube<T> &Hypercube)
{
//	this->file = Hypercube; //The hypercube is not modified afterwards

	this->set_stdmap_transpose(std_map, Hypercube.data, M.lstd, M.ustd);

}
////START
template<typename T>
algo_rohsa<T>::algo_rohsa(parameters<T> &M, hypercube<T> &Hypercube)
{
////	this->file = Hypercube; //The hypercube is not modified afterwards

//  Dimensions of data and cube
	this->dim_cube = Hypercube.get_dim_cube();
	this->dim_data = Hypercube.get_dim_data();

//	Dimensions of the cube /!\ dim_x, dim_y, dim_v stand for the spatial and spectral dimensions of the cube
	this->dim_x = dim_cube[0];
	this->dim_y = dim_cube[1];
	this->dim_v = dim_cube[2];

	this->temps_detail_regu = (double*)malloc((Nb_time_mes+1)*sizeof(double));
	this->temps_detail_regu[0] = 0.5;

	std_spectrum(Hypercube, this->dim_data[0], this->dim_data[1], this->dim_data[2]); 
	mean_spectrum(Hypercube, this->dim_data[0], this->dim_data[1], this->dim_data[2]);
	max_spectrum(Hypercube, this->dim_data[0], this->dim_data[1], this->dim_data[2]); 
	//compute the maximum of the mean spectrum
	T max_mean_spect = *std::max_element(mean_spect.begin(), mean_spect.end());
	max_spectrum_norm(Hypercube, dim_data[0], dim_data[1], dim_data[2], max_mean_spect);

	cout.precision(std::numeric_limits<T>::max_digits10);
//	std::cout<<std::setprecision(10);

	// can't define the variable in the if
//////	std::vector<std::vector<std::vector<T>>> grid_params, fit_params;

	if(M.descent){
		std::vector<std::vector<std::vector<T>>> grid_params_(3*M.n_gauss, std::vector<std::vector<T>>(dim_data[1], std::vector<T>(dim_data[0],0.)));
		std::vector<std::vector<std::vector<T>>> fit_params_(3*M.n_gauss, std::vector<std::vector<T>>(1, std::vector<T>(1,0.)));
		this->grid_params=grid_params_;
		this->fit_params=fit_params_;
	}
	else{
		std::vector<std::vector<std::vector<T>>> grid_params_(3*M.n_gauss, std::vector<std::vector<T>>(dim_data[1], std::vector<T>(dim_data[0],0.)));
		this->grid_params=grid_params_;
	}

	std::cout<<"TEST DEBUG !"<<std::endl;
	if(M.noise_map_provided){
		if(M.three_d_noise_mode){
			//3D noise code
		}else{
		std::cout << "this->dim_cube[0] = " <<this->dim_cube[0]<< std::endl;
		std::cout << "this->dim_cube[1] = " <<this->dim_cube[1]<< std::endl;
//		Hypercube.get_noise_map_from_fits(M, this->std_data_map);
		Hypercube.get_noise_map_from_DHIGLS(M, this->std_data, this->std_cube);
		}
	}else{
		std::vector<std::vector<T>> std_map_init(this->dim_data[1], std::vector<T>(this->dim_data[0],0.));
		this->std_data = std_map_init;
		this->std_cube = std_map_init;
	}


	std::cout << "BEFORE" << std::endl;
	std::cout << "this->std_cube.size() : " << this->std_cube.size() << " , " << this->std_cube[0].size() <<  std::endl;
//	std::cout << "std_data_map.size() : " << std_data_map.size() << " , " << std_data_map[0].size() <<  std::endl;
//	std::cout << "fit_params.size() : " << fit_params.size() << " , " << fit_params[0].size() << " , " << fit_params[0][0].size() <<  std::endl;
	std::cout << "grid_params.size() : " << grid_params.size() << " , " << grid_params[0].size() << " , " << grid_params[0][0].size() <<  std::endl;
	std::cout << "AFTER" << std::endl;

	if(M.descent){
		std::cout<<"START MULTIRESOLUTION"<<std::endl;
		descente(Hypercube, M, this->grid_params, this->fit_params);
	}else{
		std::cout<<"START TEST TOOLBOX !"<<std::endl;
		test_toolbox(Hypercube, M, this->grid_params);
	}

//	std::cout<<"TEST DEBUG BEFORE WRITING !"<<std::endl;
//	this->file.write_into_binary(M, this->grid_params);
//	std::cout<<"TEST DEBUG END !"<<std::endl;
}

//void algo_rohsa<T>::descente(parameters &M, std::vector<std::vector<std::vector<T>>> &grid_params, std::vector<std::vector<std::vector<T>>> &fit_params){
template <typename T>
void algo_rohsa<T>::descente(hypercube<T> &Hypercube, parameters<T> &M, std::vector<std::vector<std::vector<T>>> &grid_params, std::vector<std::vector<std::vector<T>>> &fit_params){

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

	std::cout<<"M.input_format_fits = "<<M.input_format_fits<<std::endl;
	std::cout<<"M.filename_dat = "<<M.filename_dat<<std::endl;
	std::cout<<"M.filename_fits = "<<M.filename_fits<<std::endl;
	std::cout<<"M.output_format_fits = "<<M.output_format_fits<<std::endl;
	std::cout<<"M.fileout = "<<M.fileout<<std::endl;
	std::cout<<"M.noise_map_provided = "<<M.noise_map_provided<<std::endl;
	std::cout<<"M.filename_noise = "<<M.filename_noise<<std::endl;
	std::cout<<"M.give_input_spectrum = "<<M.give_input_spectrum<<std::endl;
	printf("M.n_gauss = %d\n",M.n_gauss);
	printf("M.lambda_amp = %f\n",M.lambda_amp);
	printf("M.lambda_mu = %f\n",M.lambda_mu);
	printf("M.lambda_sig = %f\n",M.lambda_sig);
	printf("M.lambda_var_sig = %f\n",M.lambda_var_sig);
	printf("M.amp_fact_init = %f\n",M.amp_fact_init);
	printf("M.sig_init = %f\n",M.sig_init);
	printf("M.lstd = %d\n",M.lstd);
	printf("M.ustd = %d\n",M.ustd);
	printf("M.ub_sig = %f\n",M.ub_sig);
	printf("M.lb_sig = %f\n",M.lb_sig);
	printf("M.ub_sig_init = %f\n",M.ub_sig_init);
	printf("M.lb_sig_init = %f\n",M.lb_sig_init);
	printf("M.maxiter_init = %d\n",M.maxiter_init);
	printf("M.maxiter = %d\n",M.maxiter);
	printf("M.m = %d\n",M.m);
	std::cout<<"M.init_option = "<<M.init_option<<std::endl;
	std::cout<<"M.regul = "<<M.regul<<std::endl;
	std::cout<<"M.descent = "<<M.descent<<std::endl;
	std::cout<<"M.print_mean_parameters = "<<M.print_mean_parameters<<std::endl;
	printf("M.iprint = %d\n",int(M.iprint));
	printf("M.iprint_init = %d\n",int(M.iprint_init));

	if(!(M.jump_to_last_level)){
		for(n=0; n<Hypercube.nside; n++)
		{
			double temps1_init_spectrum = omp_get_wtime();

			int power = pow(2,n);

			std::cout << " power = " << power << std::endl;

			std::vector<std::vector<std::vector<T>>> cube_mean(power, std::vector<std::vector<T>>(power,std::vector<T>(dim_v,1.)));

			mean_array(Hypercube, power, cube_mean);

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
//						upgrade(M ,cube_mean, fit_params, power);
						double temps2_upgrade = omp_get_wtime();
						temps_upgrade+=temps2_upgrade-temps1_upgrade;
					}
					if (n>0 and n<Hypercube.nside){
						std::vector<std::vector<T>> std_map(power, std::vector<T>(power,0.));
						double temps_std_map1=omp_get_wtime();
						if (M.noise_map_provided){
//							reshape_noise_up(this->std_cube);
////							mean_noise_map(this->Hypercube.nside, n, power, this->std_cube, std_map);
							mean_noise_map(Hypercube.nside, n, power, this->std_cube, std_map);
							std::cout<<"Hypercube.nside = "<<Hypercube.nside<<std::endl;
							printf("std_map[0][0] = %.26f\n",std_map[0][0]);
							printf("std_map[0][1] = %.26f\n",std_map[0][1]);
							printf("std_map[1][0] = %.26f\n",std_map[1][0]);
							printf("std_map[1][1] = %.26f\n",std_map[1][1]);
					/*
							std::cout<<"std_map[1][0] = "<<std_map[1][0]<<std::endl;
							std::cout<<"std_map[0][1] = "<<std_map[0][1]<<std::endl;
							std::cout<<"std_map[1][1] = "<<std_map[1][1]<<std::endl;
							for(int i = 0; i<this->dim_data[2]; i++) std::cout<<"cube_mean[0][0]["<<i<<"] = "<<cube_mean[0][0][i]<<std::endl;
							for(int i = 0; i<this->dim_data[2]; i++) std::cout<<"cube_mean[0][1]["<<i<<"] = "<<cube_mean[0][1][i]<<std::endl;
							for(int i = 0; i<this->dim_data[2]; i++) std::cout<<"cube_mean[1][0]["<<i<<"] = "<<cube_mean[1][0][i]<<std::endl;
							for(int i = 0; i<this->dim_data[2]; i++) std::cout<<"cube_mean[1][1]["<<i<<"] = "<<cube_mean[1][1][i]<<std::endl;
//							std::cout<<"cube_mean[0][0][0] = "<<cube_mean[0][0][0]<<std::endl;
//							std::cout<<"cube_mean[1][0][0] = "<<cube_mean[1][0][0]<<std::endl;
//							std::cout<<"cube_mean[0][1][0] = "<<cube_mean[0][1][0]<<std::endl;
//							std::cout<<"cube_mean[0][0][1] = "<<cube_mean[0][0][1]<<std::endl;
					*/
						}else{
							set_stdmap_transpose(std_map, cube_mean, M.lstd, M.ustd);
							std::cout<<"std_map[0][0] = "<<std_map[0][0]<<std::endl;
							std::cout<<"std_map[1][0] = "<<std_map[1][0]<<std::endl;
							std::cout<<"std_map[0][1] = "<<std_map[0][1]<<std::endl;
							std::cout<<"cube_mean[0][0][0] = "<<cube_mean[0][0][0]<<std::endl;
							std::cout<<"cube_mean[1][0][0] = "<<cube_mean[1][0][0]<<std::endl;
							std::cout<<"cube_mean[0][1][0] = "<<cube_mean[0][1][0]<<std::endl;
							std::cout<<"cube_mean[0][0][1] = "<<cube_mean[0][0][1]<<std::endl;
	//						set_stdmap(std_map, cube_mean, M.lstd, M.ustd); //?
						}
						double temps_std_map2=omp_get_wtime();
						temps_std_map_pp+=temps_std_map2-temps_std_map1;

						double temps1_update_pp=omp_get_wtime();
/*
						std::cout<<"this->data[0][0][0] = "<<this->data[0][0][0]<<std::endl;
						std::cout<<"this->data[1][0][0] = "<<this->data[1][0][0]<<std::endl;
						std::cout<<"this->data[0][1][0] = "<<this->data[0][1][0]<<std::endl;
						std::cout<<"this->data[0][0][1] = "<<this->data[0][0][1]<<std::endl;
						exit(0);
*/
						update_clean(Hypercube, M, cube_mean, fit_params, std_map, power, power, dim_v, b_params);

						double temps2_update_pp=omp_get_wtime();
						temps_update_pp += temps2_update_pp-temps1_update_pp;

						if(M.print_mean_parameters){
							mean_parameters(fit_params);
						}
					}
				}

		double temps_go_up_level1=omp_get_wtime();
		if(M.save_grid_through_multiresolution){
			Hypercube.save_result_multires(fit_params, M, n);
		}
//		Hypercube.write_in_file(grid_params);

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
////			this->file.write_in_file(grid_params);
			Hypercube.write_in_file(grid_params);
		}

	}else{
////		this->file.get_from_file(grid_params, 3*M.n_gauss, this->dim_data[1], this->dim_data[0]);
		Hypercube.get_from_file(grid_params, 3*M.n_gauss, this->dim_data[1], this->dim_data[0]);
	}


	this->grid_params = grid_params;
	double temps_dernier_niveau1 = omp_get_wtime();
	double temps_std_map1=omp_get_wtime();
	if(!M.noise_map_provided){
////		set_stdmap(this->std_data, this->file.data, M.lstd, M.ustd);
		set_stdmap(this->std_data, Hypercube.data, M.lstd, M.ustd);
	}

	double temps_std_map2=omp_get_wtime();
	temps_std_map_dp+=temps_std_map2-temps_std_map1;

	double temps_update_dp1 = omp_get_wtime();

	std::cout<<"std_data[0][0] = "<<std_data[0][0]<<std::endl;
	std::cout<<"std_data[1][0] = "<<std_data[1][0]<<std::endl;
	std::cout<<"std_data[0][1] = "<<std_data[0][1]<<std::endl;
	std::cout<<"std_data[1][1] = "<<std_data[1][1]<<std::endl;

	if(M.regul){
		std::cout<<"Updating last level"<<std::endl;
////		update_clean(M, this->file.data, grid_params, this->std_data, this->dim_data[0], this->dim_data[1], this->dim_v, b_params);
		update_clean(Hypercube, M, Hypercube.data, grid_params, this->std_data, this->dim_data[0], this->dim_data[1], this->dim_v, b_params);
//		update_clean(M, this->file.data, grid_params, this->std_cube, this->dim_data[0], this->dim_data[1], this->dim_v, b_params);

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
	std::cout<<"                                "<<std::endl;
	if(this->temps_detail_regu[0]>1.){
		std::cout<<"            Details regularization            "<<std::endl;
		std::cout<<"                                "<<std::endl;
		for(int i = 1; i<=Nb_time_mes ; i++){
			std::cout<< "	-> "<<i<<" : " << this->temps_detail_regu[i]  <<std::endl;
		}
		std::cout<<"                                "<<std::endl;
	}
	std::cout<<"           End of details section             "<<std::endl;
	std::cout<<"                                "<<std::endl;
	std::cout<<"                                "<<std::endl;
}

template <typename T>
void algo_rohsa<T>::test_toolbox(hypercube<T> &Hypercube, parameters<T> &M, std::vector<std::vector<std::vector<T>>> &grid_params){
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

	double temps2_before_nside;
	double temps1_descente = omp_get_wtime();

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
	printf("M.lambda_var_sig = %f\n",M.lambda_var_sig);
	printf("M.amp_fact_init = %f\n",M.amp_fact_init);
	printf("M.sig_init = %f\n",M.sig_init);
	printf("M.n_gauss = %d\n",M.n_gauss);

	double temps_dernier_niveau1 = omp_get_wtime();

	double temps_std_map1=omp_get_wtime();

	int indice_x = this->dim_data[0];
	int indice_y = this->dim_data[1];
	int indice_v = this->dim_data[2];

//	std::vector<std::vector<std::vector<T>>> grid_params(3*M.n_gauss, std::vector<std::vector<T>>(indice_y, std::vector<T>(indice_x,0.)));
	std::cout<<"TEST DEBUG !"<<std::endl;

	for(int i=0; i<indice_x; i++) {
		for(int j=0; j<indice_y; j++) {
			for(int k=0; k<3*M.n_gauss; k++) {
				grid_params[i][j][k] = 1.;
			}
		}
	}
	std::vector<T> b_params(M.n_gauss,0.);
	for(int k=0; k<M.n_gauss; k++) {
		b_params[k] = 1.;
	}
	std::cout<<"TEST DEBUG !"<<std::endl;

	if(!M.noise_map_provided){
////		set_stdmap(this->std_cube, this->file.data, M.lstd, M.ustd);
		set_stdmap(this->std_cube, Hypercube.data, M.lstd, M.ustd);
	}
	std::cout<<"TEST DEBUG !"<<std::endl;

	double temps_std_map2=omp_get_wtime();
	temps_std_map_dp+=temps_std_map2-temps_std_map1;

	double temps_update_dp1 = omp_get_wtime();

	if(M.regul){
		std::cout<<"Updating last level"<<std::endl;
////		update_clean(M, this->file.data, grid_params, this->std_cube, this->dim_data[0], this->dim_data[1], this->dim_v, b_params);
		update_clean(Hypercube, M, Hypercube.data, grid_params, this->std_cube, this->dim_data[0], this->dim_data[1], this->dim_v, b_params);
	}

	double temps_update_dp2 = omp_get_wtime();
	temps_update_dp +=temps_update_dp2-temps_update_dp1;
	this->grid_params = grid_params;
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
void algo_rohsa<T>::update_clean(hypercube<T> Hypercube, parameters<T> &M, std::vector<std::vector<std::vector<T>>> &cube_avgd_or_data, std::vector<std::vector<std::vector<T>>> &params, std::vector<std::vector<T>> &std_map, int indice_x, int indice_y, int indice_v, std::vector<T> &b_params) {
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
//				double max_line = *std::max_element(cube_flat.begin(), cube_flat.end());
//				printf("max_line = %.16f\n", max_line);
//				exit(0);
				for(int p=0; p<3*M.n_gauss; p++){
					lb_3D[p][i][j]=lb_3D_flat[p]; // the indices have been inverted
					ub_3D[p][i][j]=ub_3D_flat[p]; //
				}
			}
		}
//		exit(0);

		parameters<double> M_d;
		M_d.input_format_fits = M.input_format_fits;
		M_d.filename_dat = M.filename_dat;
		M_d.filename_fits = M.filename_fits;
		M_d.output_format_fits = M.output_format_fits;
		M_d.fileout = M.fileout;
		M_d.noise_map_provided = M.noise_map_provided;
		M_d.filename_noise = M.filename_noise;
		M_d.give_input_spectrum = M.give_input_spectrum;
		M_d.n_gauss = M.n_gauss;
		M_d.lambda_amp = double(M.lambda_amp);
		M_d.lambda_mu = double(M.lambda_mu);
		M_d.lambda_sig = double(M.lambda_sig);
		M_d.lambda_var_sig = double(M.lambda_var_sig);
		M_d.amp_fact_init = double(M.amp_fact_init);
		M_d.sig_init = double(M.sig_init);
		M_d.lstd = M.lstd;
		M_d.ustd = M.ustd;
		M_d.ub_sig = double(M.ub_sig);
		M_d.lb_sig = double(M.lb_sig);
		M_d.ub_sig_init = double(M.ub_sig_init);
		M_d.lb_sig_init = double(M.lb_sig_init);
		M_d.maxiter_init = M.maxiter_init;
		M_d.maxiter = M.maxiter;
		M_d.m = M.m;
		M_d.init_option = M.init_option;
		M_d.regul = M.regul;
		M_d.descent = M.descent;
		M_d.print_mean_parameters = M.print_mean_parameters;
		M_d.iprint = M.iprint;
		M_d.iprint_init = M.iprint_init;
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

//		minimize_clean_driver(M_d, n_beta, M.m, beta, lb, ub, cube_avgd_or_data_double, std_map_, indice_x, indice_y, indice_v, cube_flattened);
		minimize_clean(M_d, n_beta, M.m, beta, lb, ub, cube_avgd_or_data_double, std_map_, indice_x, indice_y, indice_v, cube_flattened);
		if(M.select_version == 0){ //-cpu
			one_D_to_three_D_inverted_dimensions_double(beta, params, 3*M.n_gauss, indice_y, indice_x);
		}else if(M.select_version == 1 || M.select_version == 2){ //-gpu and -h
			one_D_to_three_D_same_dimensions_double(beta, params, 3*M.n_gauss, indice_y, indice_x);
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
			//this->file.simple_plot_through_regu(params, 0,0,number_plot_2D, "début");
//			//this->file.simple_plot_through_regu(params, 0,1,number_plot_2D, "début");
//			//this->file.simple_plot_through_regu(params, 0,2,number_plot_2D, "début");
			Hypercube.save_result_multires(params, M, number_plot_2D);
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
			//this->file.simple_plot_through_regu(params, 0,0,number_plot_2D, "fin");
//			//this->file.simple_plot_through_regu(params, 0,1,number_plot_2D, "fin");
//			//this->file.simple_plot_through_regu(params, 0,2,number_plot_2D, "fin");
			Hypercube.save_result_multires(params, M, number_plot_2D);
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
	
    int* nbd_bis = NULL;
    nbd_bis = (int*)malloc(n*sizeof(int));

    long* iwa = NULL;
    iwa = (long*)malloc(taille_iwa*sizeof(long));

    int* iwa_bis = NULL;
    iwa_bis = (int*)malloc(taille_iwa*sizeof(int));

	float temps_transfert_boucle = 0.;

    double* wa = NULL;
    wa = (double*)malloc(taille_wa*sizeof(double));

    long taskValue;
    long *task=&taskValue;

    integer taskValue_bis;
    integer *task_bis=&taskValue_bis;

    double factr;
    long csaveValue;

    long *csave=&csaveValue;

    double dsave[29];
    long isave[44];
    logical lsave[4];
    double pgtol;

    for(int i(0); i<taille_wa; i++) {
		wa[i]=0.;
    }
    for(int i(0); i<taille_iwa; i++) {
		iwa[i]=0;
		iwa_bis[i]=0;
    }
    for(int i(0); i<n; i++) {
		nbd[i]=0;
    }
    for(int i(0); i<29; i++) {
		dsave[i]=0.;
    }
    for(int i(0); i<44; i++) {
		isave[i]=0;
    }
    for(int i(0); i<4; i++) {
		lsave[i]=true;
    }

	double temps2_tableau_update = omp_get_wtime();

	double* g = NULL;
	g = (double*)malloc(n*sizeof(double));
    for(int i(0); i<n; i++) {
	g[i]=0.;
    }
    f=0.;

	temps_tableau_update += omp_get_wtime() - temps2_tableau_update;

//    factr = 1e+10;
//    factr = 8.90e+9;
    factr = 1e+7;
//    factr = 1e+3;
//    pgtol = 1e-10;
    pgtol = 1e-5;


    for (i__ = 0; i__ < n; i__ ++) {
        nbd[i__] = 2;
		nbd_bis[i__] = 2;
    }

    *task = (long)START;
    *task_bis = (integer)START;

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

	if (print){//dim_x >128){
		printf("dim_x = %d , dim_y = %d , dim_v = %d \n", dim_x, dim_y, dim_v);
		printf("n = %d , n_gauss = %d\n", int(n), M.n_gauss);
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

	if(false){ //dim_x == 2 ){
		double* beta_bis = NULL;
		beta_bis = (double*)malloc(n*sizeof(double));
		if(M.n_gauss == 24){
		}else if(M.n_gauss == 18){
		}else if (M.n_gauss ==12){
 beta_bis[           0 ] =    1.9259298663736759      ;
 beta_bis[           1 ] =    39.611516214268839      ;
 beta_bis[           2 ] =    4.5770042696172206      ;
 beta_bis[           3 ] =    1.4160864367915471      ;
 beta_bis[           4 ] =    39.187071681148332      ;
 beta_bis[           5 ] =    11.412904025173709      ;
 beta_bis[           6 ] =    0.0000000000000000      ;
 beta_bis[           7 ] =    47.598609087771671      ;
 beta_bis[           8 ] =    1.0140753754025620      ;
 beta_bis[           9 ] =   0.28538450102144935      ;
 beta_bis[          10 ] =    3.6724212706602488      ;
 beta_bis[          11 ] =    10.564769223725461      ;
 beta_bis[          12 ] =   0.12218901257251603      ;
 beta_bis[          13 ] =   0.28036492644872840      ;
 beta_bis[          14 ] =    2.6311518129915674      ;
 beta_bis[          15 ] =    7.2583833588547475E-002 ;
 beta_bis[          16 ] =    30.814697070780444      ;
 beta_bis[          17 ] =    2.8097350208727661      ;
 beta_bis[          18 ] =    2.6897518219718307E-003 ;
 beta_bis[          19 ] =    44.397149231499355      ;
 beta_bis[          20 ] =    4.1948429454751910      ;
 beta_bis[          21 ] =   0.24380506758467435      ;
 beta_bis[          22 ] =    44.498131278376938      ;
 beta_bis[          23 ] =    4.4958536954677983      ;
 beta_bis[          24 ] =    5.7808620312273448E-002 ;
 beta_bis[          25 ] =    44.400891688186441      ;
 beta_bis[          26 ] =    1.6534387011427218      ;
 beta_bis[          27 ] =    1.2019255822473952E-004 ;
 beta_bis[          28 ] =    24.006240338331967      ;
 beta_bis[          29 ] =    4.9852829723476173      ;
 beta_bis[          30 ] =    1.1843864891676539E-004 ;
 beta_bis[          31 ] =    24.006582051416292      ;
 beta_bis[          32 ] =    4.9852406198154968      ;
 beta_bis[          33 ] =    5.4920760639833195E-004 ;
 beta_bis[          34 ] =    24.006762136932245      ;
 beta_bis[          35 ] =    4.9854771222766878      ;
 beta_bis[          36 ] =    1.9259298663736759      ;
 beta_bis[          37 ] =    39.611516214268839      ;
 beta_bis[          38 ] =    4.5770042696172206      ;
 beta_bis[          39 ] =    1.4160864367915471      ;
 beta_bis[          40 ] =    39.187071681148332      ;
 beta_bis[          41 ] =    11.412904025173709      ;
 beta_bis[          42 ] =    0.0000000000000000      ;
 beta_bis[          43 ] =    47.598609087771671      ;
 beta_bis[          44 ] =    1.0140753754025620      ;
 beta_bis[          45 ] =   0.28538450102144935      ;
 beta_bis[          46 ] =    3.6724212706602488      ;
 beta_bis[          47 ] =    10.564769223725461      ;
 beta_bis[          48 ] =   0.12218901257251603      ;
 beta_bis[          49 ] =   0.28036492644872840      ;
 beta_bis[          50 ] =    2.6311518129915674      ;
 beta_bis[          51 ] =    7.2583833588547475E-002 ;
 beta_bis[          52 ] =    30.814697070780444      ;
 beta_bis[          53 ] =    2.8097350208727661      ;
 beta_bis[          54 ] =    2.6897518219718307E-003 ;
 beta_bis[          55 ] =    44.397149231499355      ;
 beta_bis[          56 ] =    4.1948429454751910      ;
 beta_bis[          57 ] =   0.24380506758467435      ;
 beta_bis[          58 ] =    44.498131278376938      ;
 beta_bis[          59 ] =    4.4958536954677983      ;
 beta_bis[          60 ] =    5.7808620312273448E-002 ;
 beta_bis[          61 ] =    44.400891688186441      ;
 beta_bis[          62 ] =    1.6534387011427218      ;
 beta_bis[          63 ] =    1.2019255822473952E-004 ;
 beta_bis[          64 ] =    24.006240338331967      ;
 beta_bis[          65 ] =    4.9852829723476173      ;
 beta_bis[          66 ] =    1.1843864891676539E-004 ;
 beta_bis[          67 ] =    24.006582051416292      ;
 beta_bis[          68 ] =    4.9852406198154968      ;
 beta_bis[          69 ] =    5.4920760639833195E-004 ;
 beta_bis[          70 ] =    24.006762136932245      ;
 beta_bis[          71 ] =    4.9854771222766878      ;
 beta_bis[          72 ] =    1.9259298663736759      ;
 beta_bis[          73 ] =    39.611516214268839      ;
 beta_bis[          74 ] =    4.5770042696172206      ;
 beta_bis[          75 ] =    1.4160864367915471      ;
 beta_bis[          76 ] =    39.187071681148332      ;
 beta_bis[          77 ] =    11.412904025173709      ;
 beta_bis[          78 ] =    0.0000000000000000      ;
 beta_bis[          79 ] =    47.598609087771671      ;
 beta_bis[          80 ] =    1.0140753754025620      ;
 beta_bis[          81 ] =   0.28538450102144935      ;
 beta_bis[          82 ] =    3.6724212706602488      ;
 beta_bis[          83 ] =    10.564769223725461      ;
 beta_bis[          84 ] =   0.12218901257251603      ;
 beta_bis[          85 ] =   0.28036492644872840      ;
 beta_bis[          86 ] =    2.6311518129915674      ;
 beta_bis[          87 ] =    7.2583833588547475E-002 ;
 beta_bis[          88 ] =    30.814697070780444      ;
 beta_bis[          89 ] =    2.8097350208727661      ;
 beta_bis[          90 ] =    2.6897518219718307E-003 ;
 beta_bis[          91 ] =    44.397149231499355      ;
 beta_bis[          92 ] =    4.1948429454751910      ;
 beta_bis[          93 ] =   0.24380506758467435      ;
 beta_bis[          94 ] =    44.498131278376938      ;
 beta_bis[          95 ] =    4.4958536954677983      ;
 beta_bis[          96 ] =    5.7808620312273448E-002 ;
 beta_bis[          97 ] =    44.400891688186441      ;
 beta_bis[          98 ] =    1.6534387011427218      ;
 beta_bis[          99 ] =    1.2019255822473952E-004 ;
 beta_bis[         100 ] =    24.006240338331967      ;
 beta_bis[         101 ] =    4.9852829723476173      ;
 beta_bis[         102 ] =    1.1843864891676539E-004 ;
 beta_bis[         103 ] =    24.006582051416292      ;
 beta_bis[         104 ] =    4.9852406198154968      ;
 beta_bis[         105 ] =    5.4920760639833195E-004 ;
 beta_bis[         106 ] =    24.006762136932245      ;
 beta_bis[         107 ] =    4.9854771222766878      ;
 beta_bis[         108 ] =    1.9259298663736759      ;
 beta_bis[         109 ] =    39.611516214268839      ;
 beta_bis[         110 ] =    4.5770042696172206      ;
 beta_bis[         111 ] =    1.4160864367915471      ;
 beta_bis[         112 ] =    39.187071681148332      ;
 beta_bis[         113 ] =    11.412904025173709      ;
 beta_bis[         114 ] =    0.0000000000000000      ;
 beta_bis[         115 ] =    47.598609087771671      ;
 beta_bis[         116 ] =    1.0140753754025620      ;
 beta_bis[         117 ] =   0.28538450102144935      ;
 beta_bis[         118 ] =    3.6724212706602488      ;
 beta_bis[         119 ] =    10.564769223725461      ;
 beta_bis[         120 ] =   0.12218901257251603      ;
 beta_bis[         121 ] =   0.28036492644872840      ;
 beta_bis[         122 ] =    2.6311518129915674      ;
 beta_bis[         123 ] =    7.2583833588547475E-002 ;
 beta_bis[         124 ] =    30.814697070780444      ;
 beta_bis[         125 ] =    2.8097350208727661      ;
 beta_bis[         126 ] =    2.6897518219718307E-003 ;
 beta_bis[         127 ] =    44.397149231499355      ;
 beta_bis[         128 ] =    4.1948429454751910      ;
 beta_bis[         129 ] =   0.24380506758467435      ;
 beta_bis[         130 ] =    44.498131278376938      ;
 beta_bis[         131 ] =    4.4958536954677983      ;
 beta_bis[         132 ] =    5.7808620312273448E-002 ;
 beta_bis[         133 ] =    44.400891688186441      ;
 beta_bis[         134 ] =    1.6534387011427218      ;
 beta_bis[         135 ] =    1.2019255822473952E-004 ;
 beta_bis[         136 ] =    24.006240338331967      ;
 beta_bis[         137 ] =    4.9852829723476173      ;
 beta_bis[         138 ] =    1.1843864891676539E-004 ;
 beta_bis[         139 ] =    24.006582051416292      ;
 beta_bis[         140 ] =    4.9852406198154968      ;
 beta_bis[         141 ] =    5.4920760639833195E-004 ;
 beta_bis[         142 ] =    24.006762136932245      ;
 beta_bis[         143 ] =    4.9854771222766878      ;
 beta_bis[         144 ] =    4.5770048851219984      ;
 beta_bis[         145 ] =    11.412894554444117      ;
 beta_bis[         146 ] =    1.0140753754025620      ;
 beta_bis[         147 ] =    10.564775944844422      ;
 beta_bis[         148 ] =    2.6311378727042491      ;
 beta_bis[         149 ] =    2.8097365086622692      ;
 beta_bis[         150 ] =    4.1948428861690363      ;
 beta_bis[         151 ] =    4.4958538815866556      ;
 beta_bis[         152 ] =    1.6534352918812767      ;
 beta_bis[         153 ] =    4.9852829803808287      ;
 beta_bis[         154 ] =    4.9852406277341466      ;
 beta_bis[         155 ] =    4.9854771582078241      ;
		}else if (M.n_gauss ==6){
 beta_bis[           0 ] =    2.0877938739618664      ;
 beta_bis[           1 ] =    40.080739377482168      ;
 beta_bis[           2 ] =    4.8947732102892880      ;
 beta_bis[           3 ] =    1.4031533793182884      ;
 beta_bis[           4 ] =    39.158983744693593      ;
 beta_bis[           5 ] =    11.499638699685757      ;
 beta_bis[           6 ] =    0.0000000000000000      ;
 beta_bis[           7 ] =    47.598320779838893      ;
 beta_bis[           8 ] =    1.0137391095631108      ;
 beta_bis[           9 ] =   0.28811754258399985      ;
 beta_bis[          10 ] =    3.3254846793813391      ;
 beta_bis[          11 ] =    10.510376840123156      ;
 beta_bis[          12 ] =   0.11816490310404941      ;
 beta_bis[          13 ] =   0.17081096144200053      ;
 beta_bis[          14 ] =    2.6715350987595698      ;
 beta_bis[          15 ] =    3.7937138271780163E-002 ;
 beta_bis[          16 ] =    30.161267116633155      ;
 beta_bis[          17 ] =    2.1661281678862410      ;
 beta_bis[          18 ] =    2.0877938739618664      ;
 beta_bis[          19 ] =    40.080739377482168      ;
 beta_bis[          20 ] =    4.8947732102892880      ;
 beta_bis[          21 ] =    1.4031533793182884      ;
 beta_bis[          22 ] =    39.158983744693593      ;
 beta_bis[          23 ] =    11.499638699685757      ;
 beta_bis[          24 ] =    0.0000000000000000      ;
 beta_bis[          25 ] =    47.598320779838893      ;
 beta_bis[          26 ] =    1.0137391095631108      ;
 beta_bis[          27 ] =   0.28811754258399985      ;
 beta_bis[          28 ] =    3.3254846793813391      ;
 beta_bis[          29 ] =    10.510376840123156      ;
 beta_bis[          30 ] =   0.11816490310404941      ;
 beta_bis[          31 ] =   0.17081096144200053      ;
 beta_bis[          32 ] =    2.6715350987595698      ;
 beta_bis[          33 ] =    3.7937138271780163E-002 ;
 beta_bis[          34 ] =    30.161267116633155      ;
 beta_bis[          35 ] =    2.1661281678862410      ;
 beta_bis[          36 ] =    2.0877938739618664      ;
 beta_bis[          37 ] =    40.080739377482168      ;
 beta_bis[          38 ] =    4.8947732102892880      ;
 beta_bis[          39 ] =    1.4031533793182884      ;
 beta_bis[          40 ] =    39.158983744693593      ;
 beta_bis[          41 ] =    11.499638699685757      ;
 beta_bis[          42 ] =    0.0000000000000000      ;
 beta_bis[          43 ] =    47.598320779838893      ;
 beta_bis[          44 ] =    1.0137391095631108      ;
 beta_bis[          45 ] =   0.28811754258399985      ;
 beta_bis[          46 ] =    3.3254846793813391      ;
 beta_bis[          47 ] =    10.510376840123156      ;
 beta_bis[          48 ] =   0.11816490310404941      ;
 beta_bis[          49 ] =   0.17081096144200053      ;
 beta_bis[          50 ] =    2.6715350987595698      ;
 beta_bis[          51 ] =    3.7937138271780163E-002 ;
 beta_bis[          52 ] =    30.161267116633155      ;
 beta_bis[          53 ] =    2.1661281678862410      ;
 beta_bis[          54 ] =    2.0877938739618664      ;
 beta_bis[          55 ] =    40.080739377482168      ;
 beta_bis[          56 ] =    4.8947732102892880      ;
 beta_bis[          57 ] =    1.4031533793182884      ;
 beta_bis[          58 ] =    39.158983744693593      ;
 beta_bis[          59 ] =    11.499638699685757      ;
 beta_bis[          60 ] =    0.0000000000000000      ;
 beta_bis[          61 ] =    47.598320779838893      ;
 beta_bis[          62 ] =    1.0137391095631108      ;
 beta_bis[          63 ] =   0.28811754258399985      ;
 beta_bis[          64 ] =    3.3254846793813391      ;
 beta_bis[          65 ] =    10.510376840123156      ;
 beta_bis[          66 ] =   0.11816490310404941      ;
 beta_bis[          67 ] =   0.17081096144200053      ;
 beta_bis[          68 ] =    2.6715350987595698      ;
 beta_bis[          69 ] =    3.7937138271780163E-002 ;
 beta_bis[          70 ] =    30.161267116633155      ;
 beta_bis[          71 ] =    2.1661281678862410      ;
 beta_bis[          72 ] =    4.8947744447858961      ;
 beta_bis[          73 ] =    11.499639568058166      ;
 beta_bis[          74 ] =    1.0137391095631108      ;
 beta_bis[          75 ] =    10.510377083823741      ;
 beta_bis[          76 ] =    2.6715358229273298      ;
 beta_bis[          77 ] =    2.1661290746787594      ;
		}else if (M.n_gauss ==8){
 beta_bis[           0 ] =    2.0815856425409653      ;
 beta_bis[           1 ] =    40.084691382283758      ;
 beta_bis[           2 ] =    4.8837525120574172      ;
 beta_bis[           3 ] =    1.4097636472445290      ;
 beta_bis[           4 ] =    39.161260846480587      ;
 beta_bis[           5 ] =    11.477286726786648      ;
 beta_bis[           6 ] =    0.0000000000000000      ;
 beta_bis[           7 ] =    47.598320779838893      ;
 beta_bis[           8 ] =    1.0137391095631108      ;
 beta_bis[           9 ] =   0.28969220631531623      ;
 beta_bis[          10 ] =    3.3359506631557752      ;
 beta_bis[          11 ] =    10.466427351233628      ;
 beta_bis[          12 ] =   0.11833211787345856      ;
 beta_bis[          13 ] =   0.15710146500820116      ;
 beta_bis[          14 ] =    2.6120097331325067      ;
 beta_bis[          15 ] =    3.7146746896151765E-002 ;
 beta_bis[          16 ] =    30.279331550070566      ;
 beta_bis[          17 ] =    2.3499004390826999      ;
 beta_bis[          18 ] =    4.1934115977074646E-005 ;
 beta_bis[          19 ] =    44.419006866940201      ;
 beta_bis[          20 ] =    4.2366125459249018      ;
 beta_bis[          21 ] =    1.1364212766045509E-004 ;
 beta_bis[          22 ] =    44.427028202944456      ;
 beta_bis[          23 ] =    4.2656898378758390      ;
 beta_bis[          24 ] =    2.0815856425409653      ;
 beta_bis[          25 ] =    40.084691382283758      ;
 beta_bis[          26 ] =    4.8837525120574172      ;
 beta_bis[          27 ] =    1.4097636472445290      ;
 beta_bis[          28 ] =    39.161260846480587      ;
 beta_bis[          29 ] =    11.477286726786648      ;
 beta_bis[          30 ] =    0.0000000000000000      ;
 beta_bis[          31 ] =    47.598320779838893      ;
 beta_bis[          32 ] =    1.0137391095631108      ;
 beta_bis[          33 ] =   0.28969220631531623      ;
 beta_bis[          34 ] =    3.3359506631557752      ;
 beta_bis[          35 ] =    10.466427351233628      ;
 beta_bis[          36 ] =   0.11833211787345856      ;
 beta_bis[          37 ] =   0.15710146500820116      ;
 beta_bis[          38 ] =    2.6120097331325067      ;
 beta_bis[          39 ] =    3.7146746896151765E-002 ;
 beta_bis[          40 ] =    30.279331550070566      ;
 beta_bis[          41 ] =    2.3499004390826999      ;
 beta_bis[          42 ] =    4.1934115977074646E-005 ;
 beta_bis[          43 ] =    44.419006866940201      ;
 beta_bis[          44 ] =    4.2366125459249018      ;
 beta_bis[          45 ] =    1.1364212766045509E-004 ;
 beta_bis[          46 ] =    44.427028202944456      ;
 beta_bis[          47 ] =    4.2656898378758390      ;
 beta_bis[          48 ] =    2.0815856425409653      ;
 beta_bis[          49 ] =    40.084691382283758      ;
 beta_bis[          50 ] =    4.8837525120574172      ;
 beta_bis[          51 ] =    1.4097636472445290      ;
 beta_bis[          52 ] =    39.161260846480587      ;
 beta_bis[          53 ] =    11.477286726786648      ;
 beta_bis[          54 ] =    0.0000000000000000      ;
 beta_bis[          55 ] =    47.598320779838893      ;
 beta_bis[          56 ] =    1.0137391095631108      ;
 beta_bis[          57 ] =   0.28969220631531623      ;
 beta_bis[          58 ] =    3.3359506631557752      ;
 beta_bis[          59 ] =    10.466427351233628      ;
 beta_bis[          60 ] =   0.11833211787345856      ;
 beta_bis[          61 ] =   0.15710146500820116      ;
 beta_bis[          62 ] =    2.6120097331325067      ;
 beta_bis[          63 ] =    3.7146746896151765E-002 ;
 beta_bis[          64 ] =    30.279331550070566      ;
 beta_bis[          65 ] =    2.3499004390826999      ;
 beta_bis[          66 ] =    4.1934115977074646E-005 ;
 beta_bis[          67 ] =    44.419006866940201      ;
 beta_bis[          68 ] =    4.2366125459249018      ;
 beta_bis[          69 ] =    1.1364212766045509E-004 ;
 beta_bis[          70 ] =    44.427028202944456      ;
 beta_bis[          71 ] =    4.2656898378758390      ;
 beta_bis[          72 ] =    2.0815856425409653      ;
 beta_bis[          73 ] =    40.084691382283758      ;
 beta_bis[          74 ] =    4.8837525120574172      ;
 beta_bis[          75 ] =    1.4097636472445290      ;
 beta_bis[          76 ] =    39.161260846480587      ;
 beta_bis[          77 ] =    11.477286726786648      ;
 beta_bis[          78 ] =    0.0000000000000000      ;
 beta_bis[          79 ] =    47.598320779838893      ;
 beta_bis[          80 ] =    1.0137391095631108      ;
 beta_bis[          81 ] =   0.28969220631531623      ;
 beta_bis[          82 ] =    3.3359506631557752      ;
 beta_bis[          83 ] =    10.466427351233628      ;
 beta_bis[          84 ] =   0.11833211787345856      ;
 beta_bis[          85 ] =   0.15710146500820116      ;
 beta_bis[          86 ] =    2.6120097331325067      ;
 beta_bis[          87 ] =    3.7146746896151765E-002 ;
 beta_bis[          88 ] =    30.279331550070566      ;
 beta_bis[          89 ] =    2.3499004390826999      ;
 beta_bis[          90 ] =    4.1934115977074646E-005 ;
 beta_bis[          91 ] =    44.419006866940201      ;
 beta_bis[          92 ] =    4.2366125459249018      ;
 beta_bis[          93 ] =    1.1364212766045509E-004 ;
 beta_bis[          94 ] =    44.427028202944456      ;
 beta_bis[          95 ] =    4.2656898378758390      ;
 beta_bis[          96 ] =    4.8889634377581341      ;
 beta_bis[          97 ] =    11.546297399141780      ;
 beta_bis[          98 ] =    1.0137391095631108      ;
 beta_bis[          99 ] =    10.393992417051951      ;
 beta_bis[         100 ] =    2.7083299189770411      ;
 beta_bis[         101 ] =    2.4407159743939841      ;
 beta_bis[         102 ] =    4.2396222563425825      ;
 beta_bis[         103 ] =    4.2675092545588118      ;
		}
		std::cout<<"beta_bis[          71 ] = "<<beta_bis[          71 ]<<std::endl;
		std::cout<<"M.select_version ="<<M.select_version<<std::endl;
//		std::cout<<"M.select_version "<<M.select_version<<std::endl;

 		if(M.select_version == 0){
			for(int i = 0; i<dim_x; i++){
				for(int j = 0; j<dim_y; j++){
					for(int k = 0; k<3*M.n_gauss; k++){
			beta[i*dim_y*3*M.n_gauss+j*3*M.n_gauss+k] = beta_bis[i*dim_y*3*M.n_gauss+j*3*M.n_gauss+k];
					}
				}
			}
			for(int i = 0; i<M.n_gauss; i++){
				beta[n-M.n_gauss+i] = beta_bis[n-M.n_gauss+i];
			}
 		}else{
			for(int i = 0; i<dim_x; i++){
				for(int j = 0; j<dim_y; j++){
					for(int k = 0; k<3*M.n_gauss; k++){
			beta[k*dim_y*dim_x+j*dim_x+i] = beta_bis[i*dim_y*3*M.n_gauss+j*3*M.n_gauss+k];
					}
				}
			}
			for(int i = 0; i<M.n_gauss; i++){
				beta[n-M.n_gauss+i] = beta_bis[n-M.n_gauss+i];
			}
		}
		free(beta_bis);
	}

	if(false ){ //dim_x == 2){
		double* beta_T = NULL;
		beta_T = (double*)malloc(n*sizeof(double));

		double* ub_T = NULL;
		ub_T = (double*)malloc(n*sizeof(double));

		double* lb_T = NULL;
		lb_T = (double*)malloc(n*sizeof(double));

		for(int i = 0; i<dim_x; i++){
			for(int j = 0; j<dim_y; j++){
				for(int k = 0; k<3*M.n_gauss; k++){
		beta_T[i*dim_y*3*M.n_gauss+j*3*M.n_gauss+k] = beta[i*dim_y*3*M.n_gauss+j*3*M.n_gauss+k];
//		ub_T[i*dim_y*3*M.n_gauss+j*3*M.n_gauss+k] = ub[i*dim_y*3*M.n_gauss+j*3*M.n_gauss+k];
//		lb_T[i*dim_y*3*M.n_gauss+j*3*M.n_gauss+k] = lb[i*dim_y*3*M.n_gauss+j*3*M.n_gauss+k];
				}
			}
		}
		for(int i = 0; i<dim_x; i++){
			for(int j = 0; j<dim_y; j++){
				for(int k = 0; k<3*M.n_gauss; k++){
		beta[i*dim_y*3*M.n_gauss+j*3*M.n_gauss+k] = beta_T[j*dim_y*3*M.n_gauss+i*3*M.n_gauss+k];
//		ub[i*dim_y*3*M.n_gauss+j*3*M.n_gauss+k] = ub_T[j*dim_y*3*M.n_gauss+i*3*M.n_gauss+k];
//		lb[i*dim_y*3*M.n_gauss+j*3*M.n_gauss+k] = lb_T[j*dim_y*3*M.n_gauss+i*3*M.n_gauss+k];
				}
			}
		}

		free(lb_T);
		free(ub_T);
		free(beta_T);
	}
	if(false){//true ){ //&& dim_x == 4){
		for(int i = 0; i<30; i++){
			printf("beta[%d] = %.26f\n",i,beta[i]);
		}
//		exit(0);
	}

	int exceed_one = 0;
//	while(IS_FG(*task) or *task==NEW_X or *task==START){
	while(IS_FG(*task_bis) or *task_bis==NEW_X or *task_bis==START){
		double temps_temp = omp_get_wtime();
	
//		std::cin.ignore();

		int n_bis = int(n);
		int m_bis = int(m);
//		setulb_(&n_bis, &m_bis, beta, lb, ub, nbd_bis, &f, g, &factr, &pgtol, wa, iwa_bis, task_bis, &M.iprint, csave, lsave, isave, dsave);
		setulb(&n, &m, beta, lb, ub, nbd, &f, g, &factr, &pgtol, wa, iwa, task, &M.iprint, csave, lsave, isave, dsave);

		temps_setulb += omp_get_wtime() - temps_temp;

		if(false){//dim_x<64){
//			f_g_cube_fast_unidimensional<double>(M, f, g, n, cube_flattened, cube, beta, dim_v, dim_y, dim_x, std_map_, temps);
	//		f_g_cube_fast_clean<double>(M, f, g, n, cube, beta, dim_v, dim_y, dim_x, std_map, temps);
	//		f_g_cube_cuda_L_clean<double>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened, temps, this->temps_transfert_d, this->temps_mirroirs, this->temps_detail_regu); // expérimentation gradient
		}else{
			if(M.select_version == 0){ //-cpu
//				f_g_cube_fast_clean<double>(M, f, g, n, cube, beta, dim_v, dim_y, dim_x, std_map, temps);
				f_g_cube_fast_unidimensional<double>(M, f, g, n, cube_flattened, cube, beta, dim_v, dim_y, dim_x, std_map_, temps);
			}else if(M.select_version == 1){ //-gpu
				double* x_dev = nullptr;
				double* g_dev = nullptr;
				cudaMalloc(&x_dev, n * sizeof(x_dev[0]));
				cudaMalloc(&g_dev, n * sizeof(g_dev[0]));
				checkCudaErrors(cudaMemset(g_dev, 0., n*sizeof(g_dev[0])));
				checkCudaErrors(cudaMemcpy(x_dev, beta, n*sizeof(double), cudaMemcpyHostToDevice));
				checkCudaErrors(cudaDeviceSynchronize());
//				f_g_cube_parallel_lib<double>(M, f, g_dev, int(n), x_dev, int(dim_v), int(dim_y), int(dim_x), std_map_dev, cube_flattened_dev, temps);
				checkCudaErrors(cudaDeviceSynchronize());
				checkCudaErrors(cudaMemcpy(g, g_dev, n*sizeof(double), cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaFree(x_dev));
				checkCudaErrors(cudaFree(g_dev));
				checkCudaErrors(cudaDeviceSynchronize());
			}else if(M.select_version == 2){ //-h
	//			f_g_cube_not_very_fast_clean<T>(M, f, g, n, cube, beta, dim_v, dim_y, dim_x, std_map);
				f_g_cube_cuda_L_clean<double>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened, temps, this->temps_transfert_d, this->temps_mirroirs, this->temps_detail_regu); // expérimentation gradient
			}
		}
		if(false ){ //&& dim_x == 4){
			for(int i = 0; i<n; i++){
				printf("g[%d] = %.26f\n",i,g[i]);
			}
			printf("f = %.26f\n",f);
			std::cin.ignore();
//			exit(0);
		}

		if (*task_bis==NEW_X ) {
			if (isave[33] >= M.maxiter) {
				*task_bis = STOP_ITER;
			}
			if (dsave[12] <= (fabs(f) + 1.) * 1e-10) {
				*task_bis = STOP_GRAD;
			}
		}
	}
//	std::cin.ignore();
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

	if(false){ //|| dim_x == 2){
		for(int i = 0; i<30; i++){
			std::cout<<"beta["<<i<<"] = "<<beta[i]<<std::endl;
		}
//		exit(0);
	}

	printf("----------------->temps_transfert_boucle = %f \n", temps_transfert_boucle);
	this->temps_transfert_d += temps_transfert_boucle;
}

template <typename T>
void algo_rohsa<T>::minimize_clean_driver(parameters<double> &M, long n_tilde, long m_tilde, double* beta, double* lb, double* ub, std::vector<std::vector<std::vector<double>>> &cube, double* std_map_, int dim_x, int dim_y, int dim_v, double* cube_flattened) {

	integer n = n_tilde;
	integer m = m_tilde;

    int i__1;
	int  i__c = 0;
    double d__1, d__2;

    double t1, t2, f;

    int i__;
    int taille_wa = 2*M.m*n+5*n+11*M.m*M.m+8*M.m;
    int taille_iwa = 3*n;

    integer* nbd = NULL;
    nbd = (integer*)malloc(n*sizeof(integer));
    integer* iwa = NULL;
    iwa = (integer*)malloc(taille_iwa*sizeof(integer));

	float temps_transfert_boucle = 0.;

    double* wa = NULL;
    wa = (double*)malloc(taille_wa*sizeof(double));

    long taskValue;
    long *task=&taskValue;

    double factr;
    integer csaveValue;

    integer *csave=&csaveValue;

    double dsave[29];
    integer isave[44];
    logical lsave[4];
    double pgtol;

    for(int i(0); i<taille_wa; i++) {
		wa[i]=0.;
    }
    for(int i(0); i<taille_iwa; i++) {
		iwa[i]=0;
    }
    for(int i(0); i<n; i++) {
		nbd[i]=0;
    }
    for(int i(0); i<29; i++) {
		dsave[i]=0.;
    }
    for(int i(0); i<44; i++) {
		isave[i]=0;
    }
    for(int i(0); i<4; i++) {
		lsave[i]=true;
    }

	double temps2_tableau_update = omp_get_wtime();

	double* g = NULL;
	g = (double*)malloc(n*sizeof(double));
    for(int i(0); i<n; i++) {
	g[i]=0.;
    }
    f=0.;

	temps_tableau_update += omp_get_wtime() - temps2_tableau_update;

    factr = 1e+7;
//    factr = 8.90e+9;
//    factr = 1e+7;
//    pgtol = 1e-10;
    pgtol = 1e-5;

    for (i__ = 0; i__ < n; i__ ++) {
        nbd[i__] = 2;
    }


	double temps1_f_g_cube = omp_get_wtime();

	std::vector<std::vector<double>> std_map(dim_y, std::vector<double>(dim_x,0.));
	for(int i=0; i<dim_y; i++){
		for(int j=0; j<dim_x; j++){
			std_map[i][j] = std_map_[i*dim_x+j];
		}
	}

	int compteur_iter_boucle_optim = 0;

	if (print){//dim_x >128){
		printf("dim_x = %d , dim_y = %d , dim_v = %d \n", dim_x, dim_y, dim_v);
		printf("n = %d , n_gauss = %d\n", int(n), M.n_gauss);
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

//    *task = (integer)START;
//    *task = (long)START;
    *task = START;
	L111:
	double temps_temp = omp_get_wtime();
	setulb(&n, &m, beta, lb, ub, nbd, &f, g, &factr, &pgtol, wa, iwa, task,
				&M.iprint, csave, lsave, isave, dsave);
	temps_setulb += omp_get_wtime() - temps_temp;


    if ( IS_FG(*task) ) {
		//f_g_cube_cuda_L_clean<double>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened, temps, this->temps_transfert_d, this->temps_mirroirs, this->temps_detail_regu); // expérimentation gradient
		f_g_cube_fast_unidimensional<double>(M, f, g, n, cube_flattened, cube, beta, dim_v, dim_y, dim_x, std_map_, temps);		
		goto L111;
	}

	if (*task==NEW_X ) {
		if (isave[33] >= M.maxiter) {
			*task = STOP_ITER;
		}
		if (dsave[12] <= (fabs(f) + 1.) * 1e-10) {
			*task = STOP_GRAD;
		}
		goto L111;
	}
	if (false){
/*
		printf("g[0] = %.16f\n",g[0]);
		printf("g[%d] = %.16f\n",1*dim_x*dim_y, g[1*dim_x*dim_y]);
		printf("g[%d] = %.16f\n",2*dim_x*dim_y, g[2*dim_x*dim_y]);
*/
		printf("g[0] = %.16f\n",g[0]);
		printf("g[1] = %.16f\n",1, g[1]);
		printf("g[2] = %.16f\n",2, g[2]);

		printf("-> f = %.16f\n",f);

		std::cout<<"x_current["<<0<<"] = "<<beta[0]<<std::endl;
		std::cout<<"x_current["<<1<<"] = "<<beta[1]<<std::endl;
		std::cout<<"x_current["<<2<<"] = "<<beta[2]<<std::endl;
	}

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
    lbfgsb_options.eps_f = static_cast<T>(1e-15);
    lbfgsb_options.eps_g = static_cast<T>(1e-15);
    lbfgsb_options.eps_x = static_cast<T>(1e-15);
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
//			std_map_[j*dim_x+i]=std_map[i][j];
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
	lbfgsb_options.eps_f = static_cast<T>(1e-14);
	lbfgsb_options.eps_g = static_cast<T>(1e-14);
	lbfgsb_options.eps_x = static_cast<T>(1e-14);
	lbfgsb_options.max_iteration = M.maxiter;
    lbfgsb_options.step_scaling = 1.;
	lbfgsb_options.hessian_approximate_dimension = M.m;
  	lbfgsb_options.machine_epsilon = 1e-14;
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
                                  const LBFGSB_CUDA_SUMMARY<T>& summary) {

	if(false){//dim_x<64){
//		f_g_cube_fast_clean_optim_CPU(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, temps);
//			f_g_cube_cuda_L_clean<double>(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened, temps, this->temps_transfert_d, this->temps_mirroirs, this->temps_detail_regu); // expérimentation gradient

		f_g_cube_fast_clean<T>(M, f, g, n, cube, x, dim_v, dim_y, dim_x, std_map, temps);
//		f_g_cube_cuda_L_clean(M, f, g, n,cube, beta, dim_v, dim_y, dim_x, std_map, cube_flattened, temps, this->temps_transfert_d, this->temps_mirroirs, this->temps_detail_regu); // expérimentation gradient
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
			f_g_cube_cuda_L_clean<T>(M, f, g, n,cube, x, dim_v, dim_y, dim_x, std_map, cube_flattened, temps, this->temps_transfert, this->temps_mirroirs, this->temps_detail_regu); // expérimentation gradient
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

	for(int m = 0; m<dim[0]; m++){
		for(int i = 0; i<dim[1]; i++){
			for(int j = 0; j<dim[2]; j++){
				for(int k = 0; k<2; k++){
					for(int l = 0; l<2; l++){
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

/*		printf("--------------------------------\n");
		for(int p(0); p<3*(i-1); p++){
			printf("x[%d] = %.16f\n",p,x[p]);
		}
			printf("--------------------------------\n");
            for(p=0; p<params.size();p++){
				printf("x_upgrade_c[%d] = %.16f\n", p, x[p]);
			}
			printf("--------------------------------\n");
*/

            init_bounds_double(M, line, M.n_gauss, lb, ub, false); //bool _init = false;
            minimize_spec(M,3*M.n_gauss ,M.m ,x ,lb , M.n_gauss, ub ,line, M.maxiter);
/*
            for(p=0; p<params.size();p++){
				printf("x_upgrade_c[%d] = %.16f\n", p, x[p]);
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
void algo_rohsa<T>::minimize_spec(parameters<T> &M, long n, long m, std::vector<double> &x_v, std::vector<double> &lb_v, int n_gauss_i, std::vector<double> &ub_v, std::vector<double> &line_v, int maxiter) {
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
            &M.iprint_init, csave, lsave, isave, dsave);


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
			if (isave[33] >= maxiter) {
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
void algo_rohsa<T>::minimize_spec_save(parameters<T> &M, long n, long m, std::vector<T> &x_v, std::vector<T> &lb_v, int n_gauss_i, std::vector<T> &ub_v, std::vector<T> &line_v, int maxiter) {
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
            &M.iprint_init, csave, lsave, isave, dsave);
//ùù
/*     if (s_cmp(task, "FG", (ftnlen)2, (ftnlen)2) == 0) { */
		if ( IS_FG(*task) ) {
			myresidual_double(x, line, _residual_, n_gauss_i);
			f = myfunc_spec_double(_residual_);
			mygrad_spec_double(g, _residual_, x, n_gauss_i);
		}


		if (*task==NEW_X ) {
			if (isave[33] >= maxiter) {
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
//	double g = 0.;
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
/*
		printf("--------------------------------\n");
		for(int p(0); p<3*(i); p++){
			printf("x_init_spec_c[%d] = %.16f\n",p,x[p]);
		}
		printf("--------------------------------\n");
*/
		minimize_spec(M, 3*i, M.m, x, lb, i, ub, line, M.maxiter_init);
/*
		for(int p(0); p<3*(i); p++){
			printf("x_init_spec_c[%d] = %.16f\n",p,x[p]);
		}
*/
		for(int p(0); p<3*(i); p++){
			params[p]=x[p];
		}
	}

}




template <typename T>
void algo_rohsa<T>::mean_array(hypercube<T> &Hypercube, int power, std::vector<std::vector<std::vector<T>>> &cube_mean)
{
	std::vector<T> spectrum(Hypercube.dim_cube[2],0.);
	int n = Hypercube.dim_cube[1]/power;
	for(int i(0); i<cube_mean[0].size(); i++)
	{
		for(int j(0); j<cube_mean.size(); j++)
		{
			for(int k(0); k<n; k++)
			{
				for (int l(0); l<n; l++)
				{
					for(int m(0); m<Hypercube.dim_cube[2]; m++)
					{
						spectrum[m] += Hypercube.cube[l+j*n][k+i*n][m];
					}
				}
			}
			for(int m(0); m<Hypercube.dim_cube[2]; m++)
			{
				cube_mean[j][i][m] = spectrum[m]/pow(n,2);
			}
			for(int p(0); p<Hypercube.dim_cube[2]; p++)
			{
				spectrum[p] = 0.;
			}
		}
	}
}


template<typename T>
void algo_rohsa<T>::reshape_noise_up(std::vector<std::vector<T>>& std_cube)
{
	int pos_x = 730;
	int pos_y = 650;
	int d_cube[] = {0,0};
	int d_map[] = {0,0};
	d_cube[0] = std_cube.size();
	d_cube[1] = std_cube[0].size();
	d_map[0] = this->std_data_map.size();
	d_map[1] = this->std_data_map[0].size();
	//compute the offset so that the data file lies in the center of a cube
//	int offset_x = (-d_cube[1]+d_map[1])/2;
//	int offset_y = (-d_cube[0]+d_map[0])/2;

	int offset_x = pos_x;
	int offset_y = pos_y;
	int half_size_0 = d_cube[0]/2;
	int half_size_1 = d_cube[1]/2;

	std::cout << "d_cube[0] = " <<d_cube[0]<< std::endl;
	std::cout << "d_cube[1] = " <<d_cube[1]<< std::endl;
	std::cout << "d_map[0] = " <<d_map[0]<< std::endl;
	std::cout << "d_map[1] = " <<d_map[1]<< std::endl;

	for(int j=offset_y; j<d_cube[0]+offset_y; j++)
	{
		for(int i=offset_x; i< d_cube[1]+offset_x; i++)
		{
			std_cube[j-offset_y][i-offset_x]= this->std_data_map[j-half_size_0][i-half_size_1];
		}
	}
/*
	printf("std_cube[0][1] = %f\n",std_cube[0][1]);
	printf("std_cube[1][0] = %f\n",std_cube[1][0]);
	printf("std_cube[1][1] = %f\n",std_cube[1][1]);
	printf("std_cube[2][0] = %f\n",std_cube[2][0]);
	printf("std_cube[0][2] = %f\n",std_cube[0][2]);
	printf("std_cube[2][2] = %f\n",std_cube[2][2]);
	std::cout<<"TEST !"<<std::endl;
	exit(0);
*/
}


template <typename T>
void algo_rohsa<T>::mean_noise_map(int n_side, int n, int power, std::vector<std::vector<T>> &std_cube, std::vector<std::vector<T>> &std_map)
{
	int dim_x = std_map[0].size();
	int dim_y = std_map.size();
	int n_macro = std_cube[0].size() / power;
	std::cout << "dim_x = " <<dim_x<< std::endl;
	std::cout << "dim_y = " <<dim_y<< std::endl;
	std::cout << "n = " <<n<< std::endl;
	std::cout << "n_macro = " <<n_macro<< std::endl;

	for(int i=0; i<dim_y; i++)
	{
		for(int j=0;j<dim_x;j++)
		{
			T val = 0.;
			for(int k=0; k<n_macro; k++)
			{
				for(int l=0;l<n_macro;l++)
				{
					val+=pow(std_cube[k+i*n_macro][l+j*n_macro],2);
// 					pow(x,y)=x^y
				}
			}
			//RMS/sqrt(N) : sqrt(sum(x(i)**2)/N)/sqrt(N) = sqrt(sum(x(i)**2))/N
			//N = pow(2,2*(n_side-n))
			//Number of spectra within the macropixel : pow(2,2*(n_side-n))
			std_map[i][j] = sqrt(val)/(pow(2,2*(n_side-n)));//pow(n,2.);
//			std_map = sqrt(val)/(2**(2*(n_side-n))))
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
void algo_rohsa<T>::std_spectrum(hypercube<T> &Hypercube, int dim_x, int dim_y, int dim_v)
{
	for(int i(0); i<dim_v; i++)
	{
		std::vector<std::vector<T>> map(dim_x, std::vector<T>(dim_y,0.));
		for(int j(0); j< dim_y; j++)
		{
			for( int k(0); k<dim_x; k++)
			{
				map[k][j] = Hypercube.data[k][j][i];
			}
		}
		std_spect.vector<T>::push_back(std_2D(map, dim_y, dim_x));
	}
}


template <typename T>
void algo_rohsa<T>::mean_spectrum(hypercube<T> &Hypercube, int dim_x, int dim_y, int dim_v)
{
	for(int i(0);i<dim_v;i++)
	{
		std::vector<std::vector<T>> map(dim_x, std::vector<T>(dim_y, 0.));
		for(int j(0); j<dim_y ; j++)
		{
			for(int k(0); k<dim_x ; k++)
			{
				map[k][j] = Hypercube.data[k][j][i];
			}
		}
		mean_spect.vector<T>::push_back(mean_2D(map, dim_y, dim_x));
		map.clear();
	}
}


template <typename T>
void algo_rohsa<T>::max_spectrum(hypercube<T> &Hypercube, int dim_x, int dim_y, int dim_v)
{
	for(int i(0); i<dim_v; i++)
	{
		std::vector<std::vector<T>> map(dim_x, std::vector<T>(dim_y,0.));
		for(int j(0); j< dim_y; j++)
		{
			for( int k(0); k<dim_x; k++)
			{
				map[k][j] = Hypercube.data[k][j][i];
			}
		}
		max_spect.vector<T>::push_back(max_2D(map, dim_y, dim_x));
		map.clear();
	}
}

template <typename T>
void algo_rohsa<T>::max_spectrum_norm(hypercube<T> &Hypercube, int dim_x, int dim_y, int dim_v, T norm_value)
{
	for(int i(0); i<dim_v; i++)
	{
		std::vector<std::vector<T>> map(dim_x, std::vector<T>(dim_y));
		for(int j(0); j< dim_y; j++)
		{
			for( int k(0); k<dim_x; k++)
			{
				map[k][j] = Hypercube.data[k][j][i];
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