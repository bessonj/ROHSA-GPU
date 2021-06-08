#ifndef DEF_FORTRAN_CONV
#define DEF_FORTRAN_CONV

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include "parameters.hpp"
#include "f_g_cube.hpp"

//This converts a C++ string (namely cstring) into a Fortran string (namely fstring). The strings formats are respectively character*60 and char* (char string[60]). 
void ConvertToFortran(char* fstring, std::size_t fstring_len, const char* cstring);
void minimize_fortran_test(parameters<double> &M, long n, long m, double* beta, double* lb, double* ub, std::vector<std::vector<std::vector<double>>> &cube, double* std_map_, int dim_x, int dim_y, int dim_v, double* cube_flattened);
int minimize();
int minimize_float();

extern "C" void square_(int *n, int *out);
extern "C" void print_a_real_(double *f);
extern "C" void print_an_array_(int *n, double *tab);
extern "C" void print_a_char_(char *char_test);
extern "C" void print_a_logical_(int *logical_test);
//extern "C" void setulb_(int *n, int *m, double* __restrict__ x, double* __restrict__ l, double* __restrict__ u, int* nbd, double* f, double* __restrict__ g, double* factr, double* pgtol, double* __restrict__ wa, int* __restrict__ iwa, char task[], int *iprint, char *csave, int *lsave, int *isave, double *dsave);
extern "C" void setulb_(int *n, int *m, double* x, double* l, double* u, int* nbd, double* f, double* g, double* factr, double* pgtol, double* wa, int* iwa, char task[], int *iprint, char *csave, int *lsave, int *isave, double *dsave);
//extern "C" void setulb_(int *n, int *m, float* x, float* l, float* u, int* nbd, float* f, float* g, float* factr, float* pgtol, float* wa, int* iwa, char task[], int *iprint, char *csave, int *lsave, int *isave, float *dsave);


#endif