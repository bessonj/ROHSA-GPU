#ifndef DEF_PARSE
#define DEF_PARSE

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <string>
#include <fstream>
#include <valarray>
#include <CCfits/CCfits>
#include <vector>

// mettre des const à la fin des déclarations si on ne modifie pas l'objet i.e. les attributs

class Parse
{
	public:

	Parse();
	Parse(std::string filename_user);
	int dim2nside();
	std::vector<int> get_dimensions_from_fits();
	void brute_show(const std::vector<std::vector<std::vector<double>>> &z, int depth, int length1, int length2);
	void multiresolution(int nside);
	int get_binary_from_fits();
	void get_vector_from_binary(std::vector<std::vector<std::vector<double>>> &z);
	void show(const std::vector<std::vector<std::vector<double>>> &z);


	private:

	std::string filename;
	std::vector<int> tab;
	int dim_x;
	int dim_y;
	int dim_v;
	std::vector<std::vector<std::vector<double>>> data;
       
};

#endif


