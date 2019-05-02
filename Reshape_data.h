#ifndef DEF_TOOL
#define DEF_TOOL

#include "Parse.h"

class Reshape_data
{
	public:

	Reshape_data(const Parse &file);
	void convolution_2D_mirror(const std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &conv, int dim_y, int dim_x, const std::vector< std::vector< double >> &kernel, int dim_k);
	void ravel_2D(const std::vector<std::vector<double>> &map, std::vector<double> &vector, int dim_y, int dim_x);
	void ravel_3D(const std::vector<std::vector<std::vector<double>>> &cube, std::vector<double> &vector, int dim_v, int dim_y, int dim_x);
	void ravel_3D_abs(const std::vector<std::vector<std::vector<double>>> &cube, const std::vector<std::vector<std::vector<double>>> &cube_abs, std::vector<double> &vector, int dim_v, int dim_y, int dim_x);
	void unravel_3D(const std::vector<double> &vector, std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x);
	void unravel_3D_abs(const std::vector<double> &vector, std::vector<std::vector<std::vector<double>>> &cube_abs,std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x);
	double Std(const std::vector<double> &array);
	double mean(const std::vector<double> &array);
	double std_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x);
	double max_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x);
	double mean_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x);
	void std_spectrum(std::vector<double> &spectrum, int dim_x, int dim_y, int dim_v);
	void reshape_up(std::vector<std::vector<std::vector<double>>> &cube);


	private:

	std::vector<std::vector<double>> kernel;
	std::vector<int> dim_data;

};

#endif



