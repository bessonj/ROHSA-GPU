#ifndef DEF_FORTRAN_CONV
#define DEF_FORTRAN_CONV

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>

//This converts a C++ string (namely cstring) into a Fortran string (namely fstring). The strings formats are respectively character*60 and char* (char string[60]). 
void ConvertToFortran(char* fstring, std::size_t fstring_len, const char* cstring);
int minimize();

extern "C" void square_(int *n, int *out);
extern "C" void print_a_real_(double *f);
extern "C" void print_an_array_(int *n, double *tab);
extern "C" void print_a_char_(char *char_test);
extern "C" void print_a_logical_(int *logical_test);
extern "C" void setulb_(int *n, int *m, double* x, double* l, double* u, int* nbd, double* f, double* g, double* factr, double* pgtol, double* wa, int* iwa, char task[], int *iprint, char *csave, int *lsave, int *isave, double *dsave);


#endif