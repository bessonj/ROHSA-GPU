#ifndef DEF_ALGO_ROHSA
#define DEF_ALGO_ROHSA

//#include "../lbfgsb-gpu/culbfgsb/culbfgsb.h"
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


class algo_rohsa
{
	public:

	algo_rohsa(parameters &M, hypercube &Hypercube); //constructeur

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

	template <typename T> void descente(parameters &M, std::vector<std::vector<std::vector<T>>> &grid_params, std::vector<std::vector<std::vector<T>>> &fit_params); //!< main loop for the multiresolution process

//	void descente(parameters &M, std::vector<std::vector<std::vector<double>>> &grid_params, std::vector<std::vector<std::vector<double>>> &fit_params); //!< main loop for the multiresolution process
/**
 * @brief Similar to descente() but without regularization.
 * 
 */

	void descente_sans_regu(parameters &M, std::vector<std::vector<std::vector<double>>> &grid_params, std::vector<std::vector<std::vector<double>>> &fit_params); //!<  main loop for the multiresolution without convolutions

	void convolution_2D_mirror(const parameters &M, const std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &conv, int dim_y, int dim_x, int dim_k); //!< 2D convolutions needed for the regularization (for std::vectors)
	void convolution_2D_mirror_flat(const parameters &M, double* image, double* &conv, int dim_y, int dim_x, int dim_k, float temps_transfert, float temps_mirroirs); //!< 2D convolutions needed for the regularization (for double flattened arrays)
	void convolution_2D_mirror_flat(const parameters &M, float* image, float* &conv, int dim_y, int dim_x, int dim_k, float temps_transfert, float temps_mirroirs); //!< 2D convolutions needed for the regularization (for float flattened arrays)

	void ravel_2D(const std::vector<std::vector<double>> &map, std::vector<double> &vector, int dim_y, int dim_x);
	void ravel_3D(const std::vector<std::vector<std::vector<double>>> &cube, double vector[], int dim_v, int dim_y, int dim_x);
	void ravel_3D(const std::vector<std::vector<std::vector<double>>> &cube, std::vector<double> &vector, int dim_v, int dim_y, int dim_x);
	void three_D_to_one_D(const std::vector<std::vector<std::vector<double>>> &cube, std::vector<double> &vector, int dim_x, int dim_y, int dim_v);
	void three_D_to_one_D(const std::vector<std::vector<std::vector<double>>> &cube, double* vector, int dim_x, int dim_y, int dim_v);
	void three_D_to_one_D_same_dimensions(const std::vector<std::vector<std::vector<double>>> &cube, double* vector, int dim_x, int dim_y, int dim_v);
	void one_D_to_three_D_same_dimensions(double* vector, std::vector<std::vector<std::vector<double>>> &cube_3D, int dim_x, int dim_y, int dim_v);
	void initialize_array(double* array, int size, double value);

	void ravel_3D_bis(const std::vector<std::vector<std::vector<double>>> &cube, double vector[], int dim_v, int dim_y, int dim_x);
	void ravel_3D_abs(const std::vector<std::vector<std::vector<double>>> &cube, const std::vector<std::vector<std::vector<double>>> &cube_abs, std::vector<double> &vector, int dim_v, int dim_y, int dim_x);
	void unravel_3D(const std::vector<double> &vector, std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x);
	void unravel_3D(double vector[], std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x);
	void unravel_3D_with_formula_transpose_xy(double vector[], std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x);
	void unravel_3D_T(double vector[], std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x);
	void unravel_3D_abs(const std::vector<double> &vector, std::vector<std::vector<std::vector<double>>> &cube_abs,std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x);
	double Std(const std::vector<double> &array);
	double mean(const std::vector<double> &array);
	double std_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x);
	double max_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x);
	double mean_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x);
	void std_spectrum(int dim_x, int dim_y, int dim_v);
	void mean_spectrum(int dim_x, int dim_y, int dim_v);
	void max_spectrum(int dim_x, int dim_y, int dim_v);
	void max_spectrum_norm(int dim_x, int dim_y, int dim_v, double norm_value);
	void init_bounds(parameters &M, std::vector<double> line, int n_gauss_local, std::vector<double> &lb, std::vector<double> &ub, bool _init); //!< Sets boundary conditions for spectrum

	void mean_array(int power, std::vector<std::vector<std::vector<double>>> &mean_array_); //!< Computes a piecewise spatially averaged array for multiresolution

	void init_spectrum(parameters &M, std::vector<double> &line, std::vector<double> &params); //!< Initializes spectrum (called during first iteration) 

	double model_function(int x, double a, double m, double s); //!< returns exp(-(x-m)^2/(2*s^2))

	int minloc(std::vector<double> &tab); //!< argmin function for a std::vector type array

	void minimize_spec(parameters &M, long n, long m, std::vector<double> &x_v, std::vector<double> &lb_v, int n_gauss_i,std::vector<double> &ub_v, std::vector<double> &line_v); //!< Solves the optimization problem during the first iteration, it calls the L-BFGS-B black box.
	void minimize_spec_save(parameters &M, long n, long m, std::vector<double> &x_v, std::vector<double> &lb_v, int n_gauss_i,std::vector<double> &ub_v, std::vector<double> &line_v); //!< Solves the optimization problem during the first iteration, it calls the L-BFGS-B black box.

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
 
	void myresidual(double params[], double line[], std::vector<double> &residual, int n_gauss_i); //!< Computes the residual along a spatial position (for a double array)
	void myresidual(std::vector<double> &params, std::vector<double> &line, std::vector<double> &residual, int n_gauss_i); //!< Computes the residual along a spatial position (for an std::vector)

	void tab_from_1Dvector_to_double(std::vector<double> vect);
	double myfunc_spec(std::vector<double> &residual);

	void mygrad_spec(double gradient[], std::vector<double> &residual, double params[], int n_gauss_i); //!< Gradient for first iteration (called in minimize_spec) 

	void upgrade(parameters &M, std::vector<std::vector<std::vector<double>>> &cube, std::vector<std::vector<std::vector<double>>> &params, int power); //!< Prepares boundary conditions and calls the minimize function. This function is used for the 1D case or if regularization is not activated. 

	void go_up_level(std::vector<std::vector<std::vector<double>>> &fit_params); //!< Projects the solution to the next resolution level.

	template <typename T> void set_stdmap(std::vector<std::vector<T>> &std_map, std::vector<std::vector<std::vector<T>>> &cube_or_data, int lb, int ub);
//	void set_stdmap(std::vector<std::vector<double>> &std_map, std::vector<std::vector<std::vector<double>>> &cube, int lb, int ub); //!< Computes the standard deviation map for every spatial position.
	template <typename T> void set_stdmap_transpose(std::vector<std::vector<T>> &std_map, std::vector<std::vector<std::vector<T>>> &cube_or_data, int lb, int ub);
	
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
	template <typename T> void update_clean(parameters &M, std::vector<std::vector<std::vector<T>>> &cube_avgd_or_data, std::vector<std::vector<std::vector<T>>> &params, std::vector<std::vector<T>> &std_map, int indice_x, int indice_y, int indice_v,std::vector<T> &b_params);//!< Prepares boundary conditions and calls the minimize function.

	template <typename T> void minimize_clean(parameters &M, long n, long m, T* beta, T* lb, T* ub, std::vector<std::vector<std::vector<T>>> &cube, std::vector<std::vector<T>> &std_map, int dim_x, int dim_y, int dim_v, T* cube_flattened);
	template <typename T> void minimize_clean_cpu(parameters &M, long n, long m, T* beta, T* lb, T* ub, std::vector<std::vector<std::vector<T>>> &cube, std::vector<std::vector<T>> &std_map, int dim_x, int dim_y, int dim_v, T* cube_flattened);
	template <typename T> void minimize_clean_gpu(parameters &M, long n, long m, T* beta, T* lb, T* ub, std::vector<std::vector<std::vector<T>>> &cube, std::vector<std::vector<T>> &std_map, int dim_x, int dim_y, int dim_v, T* cube_flattened);

	template <typename T> void f_g_cube_fast(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map);	
	template <typename T> void f_g_cube_fast_clean(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map);
	template <typename T> void f_g_cube_fast_clean_optim_CPU(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map);
	template <typename T> void f_g_cube_fast_clean_optim_CPU_lib(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T** assist_buffer);
	template <typename T> void f_g_cube_fast_without_regul(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map);
	template <typename T> void f_g_cube_omp(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map);
	template <typename T> void f_g_cube_omp_without_regul(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map);
	template <typename T> void f_g_cube_vector(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map);

	template <typename T> void f_g_cube_cuda_L(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened);
	template <typename T> void f_g_cube_cuda_L_clean(parameters &M, T &f, T g[], int n, std::vector<std::vector<std::vector<T>>> &cube, T beta[], int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened);
	template <typename T> void f_g_cube_cuda_L_clean_lib(parameters &M, T &f, T* g, int n, T* beta, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened);
	template <typename T> void f_g_cube_cuda_L_clean_lib_device_fg(parameters &M, T &f, T* g_dev, int n, T* beta_dev, int indice_v, int indice_y, int indice_x, std::vector<std::vector<T>> &std_map, T* cube_flattened);

	template <typename T> void reshape_down(std::vector<std::vector<std::vector<T>>> &tab1, std::vector<std::vector<std::vector<T>>>&tab2);

	void mean_parameters(std::vector<std::vector<std::vector<double>>> &params); //!< This routine prints the mean value of each of the \f$ 3 n\_gauss \f$ gaussian parameters

	std::vector<std::vector<std::vector<double>>> grid_params; //!< 3D array containing the gaussian parameters \f$\lambda, \mu, \sigma \f$ depending on the spatial position. Dimensions : It is a \f$ 3 n\_gauss \times dim\_y \times dim\_x \f$.
	std::vector<std::vector<std::vector<double>>> fit_params; //!< same as grid_params (gaussian parameters) but this array is used through multiresolution. Dimensions : \f$ 3 n\_gauss \times 2^k \times 2^k \f$ for \f$ 0 < k < n\_side \f$.

	double* cube_or_dat_flattened; //!< Data flattened at each iteration. Because of the multiresolution process, it has to be computed at each iteration.

//	Computationnal tools
	private:

	std::vector<std::vector<double>> kernel; //!< Kernel for convolution
	std::vector<int> dim_cube; //!< array containing the dimensions of the hypercube of spatial dimensions \f$ 2^n\_side \times 2^n\_side \f$ (for the multiresolution process) 
	std::vector<int> dim_data; //!< array containing the dimensions of the hypercube

	int dim_x;
	int dim_y;
	int dim_v;
	hypercube file; //!< Dummy 

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
	double temps_ravel;
	double temps_tableau_update;
	double temps_setulb;
	double temps_transfert_d;

	float temps_transfert;
	float temps_mirroirs;
	double temps_copy;

	int n_gauss_add; 

	std::vector<double> std_spect, mean_spect, max_spect, max_spect_norm;
	
	

/*
	int n_gauss_add;
	int nside;
	int n;
	int power;
	double ub_sig_init;
	double ub_sig;

	std::vector<int> dim_data;
	std::vector<int> dim_cube; 

*/

};


#endif
