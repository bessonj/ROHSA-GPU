#include "parameters.hpp"
#include <vector>
#include <omp.h>
#include <cmath>

void callback_test(parameters<double> &M, double &f, double g[], int n, std::vector<std::vector<std::vector<double>>> &cube, double beta[], int* dim, std::vector<std::vector<double>> &std_map, double** assist_buffer);
void three_D_to_one_D_same_dimensions(const std::vector<std::vector<std::vector<double>>> &cube_3D, double* vector, int dim_x, int dim_y, int dim_v);
void one_D_to_three_D_same_dimensions(double* vector, std::vector<std::vector<std::vector<double>>> &cube_3D, int dim_x, int dim_y, int dim_v);
void myresidual(double params[], double line[], std::vector<double> &residual, int n_gauss_i);
void myresidual(std::vector<double> &params, std::vector<double> &line, std::vector<double> &residual, int n_gauss_i);
double model_function(int x, double a, double m, double s);
void convolution_2D_mirror(const parameters<double> &M, const std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &conv, int dim_y, int dim_x, int dim_k);
double myfunc_spec(std::vector<double> &residual);