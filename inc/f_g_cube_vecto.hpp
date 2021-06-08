#include <vector>
#include <omp.h>
#include <stdio.h>
#include "gradient.hpp"
#include "convolutions.hpp"

void f_g_cube_fast_unidimensional_no_template(parameters<double> &M, double &f, double* __restrict__ g, int n, double* __restrict__ cube, std::vector<std::vector<std::vector<double>>>& __restrict__ cube_for_cache, double* __restrict__ beta, int indice_v, int indice_y, int indice_x, double* __restrict__ std_map, double* temps);
void convolution_2D_mirror_flat_vecto(const parameters<double> &M, double* __restrict__ image, double* __restrict__ conv, int dim_y, int dim_x, int dim_k);
