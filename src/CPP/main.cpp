#include "hypercube.hpp"
#include "parameters.hpp"
#include "algo_rohsa.hpp"
//#include "gradient.hpp"
#include <omp.h>
#include <cmath>
#include <iostream>
#include <string.h>
#include <stdio.h>

#include "fortran_to_cpp_conversion.h"

////VARIABLES TO BE DEFINED BY USER IF REQUIRED////
////LEAVE BLANK IF YOU ARE ALREADY USING A PRE-PROCESSED DAT FILE////

//PRE-PROCESSING OF THE FITS IMAGE
//(defines pre-processing parameters for the fits image)

//--> Choose between 2 modes :
// Either extracting a square centered on a given pixel 
// whose size is specified below or using the whole cube
// First mode :

#define EXTRACT_SQUARE_CENTERED_ON_PIX false
//~>if EXTRACT_SQUARE_CENTERED_ON_PIX is false, 
//  leave other lines below with dummy values

	#define SQUARE_SIZE 1024
	#define PIXEL_AUTOMATICALLY_CENTERED false //
//	#define PIX_X 762
	#define PIX_X 512
	#define PIX_Y 512
	//Spectral range :
	#define MIN_INDEX_RANGE_CHANNEL 0
	#define MAX_INDEX_RANGE_CHANNEL 47
	//~>end


/*
	#define SQUARE_SIZE 1024
	#define PIXEL_AUTOMATICALLY_CENTERED false //
//	#define PIX_X 762
	#define PIX_X 730
	#define PIX_Y 650
	//Spectral range :
	#define MIN_INDEX_RANGE_CHANNEL 76
	#define MAX_INDEX_RANGE_CHANNEL 123
	//~>end
*/

/*
	#define SQUARE_SIZE 256
	#define PIXEL_AUTOMATICALLY_CENTERED false //
	#define PIX_X 882
	#define PIX_Y 840
	//Spectral range :
	#define MIN_INDEX_RANGE_CHANNEL 0
	#define MAX_INDEX_RANGE_CHANNEL 149
	//~>end
*/
/*
	#define SQUARE_SIZE 2048
	#define PIXEL_AUTOMATICALLY_CENTERED false //
	#define PIX_X 1024
	#define PIX_Y 1024
	//Spectral range :
	#define MIN_INDEX_RANGE_CHANNEL 0
	#define MAX_INDEX_RANGE_CHANNEL 219
	//~>end
*/
// Second mode :
#define USE_WHOLE_FITS_FILE false


////OTHER OPTIONS 

//If true : stores the std_map in a fits file.
//(provided it's computed on the fly)
#define SAVE_STD_MAP_IN_FITS false 

//Activate 3D noise mode
#define THREE_D_NOISE_MODE false



////MACRO
#define INDEXING_2D(t,x,y) t[y+(t##_SHAPE1)*x]
#define INDEXING_3D(t,x,y,z) t[(t##_SHAPE2)*(t##_SHAPE1)*x+(t##_SHAPE2)*y+z]

double temps_test = 0;
//extern void test1(int n);

/** \mainpage Main Page
 *
 * \section comp Compiling and running the program
 *
 * We use CMake which is a cross-platform solftware managing the build process. First, create a build directory in ROHSA-GPU and go inside this directory.
 *
 * mkdir build && cd build
 *
 * Using CMake we can produce a Makefile. We may run it using make.
 *
 * cmake ../ && make
 *
 * When launching the program, we need to specify the parameters.txt file filled in by the user :  
 *
 * ./ROHSA-GPU parameters.txts
 *
 *
 * \section main main.cpp code
 *
 * \subsection par_txt Reading the user file parameters.txt
 *
 * We declare an parameters-type object that will the parameters.txt file. 
 *
 * parameters user_parametres(argv[1]);
 *
 *
 * \subsection hyp Getting the hypercube. 
 *
 * We can read the FITS or DAT file by using the class hypercube. 
 *
 * hypercube Hypercube_file(user_parametres, user_parametres.slice_index_min, user_parametres.slice_index_max, whole_data_in_cube); 
 *
 *
 * \subsection gauss_par Getting the gaussian parameters (call to the ROHSA algorithm). 
 *
 * The class algo_rohsa runs the ROHSA algorithm based on the two objects previously declared.
 *
 * algo_rohsa algo(user_parametres, Hypercube_file);
 *
 *
 * \subsection plot Plotting and storing the results.
 *
 * We can plot the smooth gaussian parameters maps and store the results back into a FITS file by using some of the routines of the class hypercube.
 *
 * algo_rohsa algo(user_parametres, Hypercube_file);
 *
 */
/// Main function : it processes the FITS file, reads the parameters.txt, solves the optimization problem through the ROHSA algo and stores/print the result.
/// 
/// Details : 2 cases are distinguished : The data file is either a *.dat or a *.fits file.

template <typename T> 
void main_routine(parameters<T> &user_parameters){

	/*
	////RECOVERS THE FITS FILE FROM THE RAW ONE
	hypercube<T> Hypercube_file(1,1,1);
	std::vector<T> newVector = Hypercube_file.read_vector_from_file("DHIGLS_UM_Tb_gauss_run_c_1024_hybrid_lambda_gauss_8_lambda_0dot1.raw");
	printf("newVector.size() = %d\n", newVector.size());
	int dim_0 = 24;
	int dim_1 = 1024;
	int dim_2 = 1024;
	std::vector <std::vector<std::vector<T>>> grid(dim_0, std::vector<std::vector<T>>(dim_1, std::vector<T>(dim_2,0.)));
	for(int k=0; k<dim_0; k++)
	{
		for(int j=0; j<dim_1; j++)
		{
			for(int i=0; i<dim_2; i++)
			{
				grid[k][j][i] = newVector[k*dim_2*dim_1+j*dim_2+i];
			}
		}
	}
	printf("before saving !\n");

	Hypercube_file.save_grid_in_fits(user_parameters, grid);
	exit(0);
	*/
/*
    hypercube<T> Hypercube_file(user_parameters, user_parameters.slice_index_min, user_parameters.slice_index_max); 
*/
	user_parameters.slice_index_min = MIN_INDEX_RANGE_CHANNEL;
	user_parameters.slice_index_max = MAX_INDEX_RANGE_CHANNEL;
	user_parameters.save_noise_map_in_fits = SAVE_STD_MAP_IN_FITS;
	user_parameters.three_d_noise_mode = THREE_D_NOISE_MODE;
//	user_parameters.size_side_square = SQUARE_SIZE;

	printf("Reading the cube ...\n");
	if (user_parameters.input_format_fits){
//		hypercube<T> Hypercube_file(user_parameters, user_parameters.slice_index_min, user_parameters.slice_index_max, PIX_X, PIX_Y, 0,0);
		hypercube<T> Hypercube_file(user_parameters, user_parameters.slice_index_min, user_parameters.slice_index_max, PIX_X, PIX_Y, SQUARE_SIZE,0);//functionning for UM

//		hypercube<T> Hypercube_file(user_parameters, user_parameters.slice_index_min, user_parameters.slice_index_max, 730, 650, SQUARE_SIZE, false, true, false);
//		hypercube<T> Hypercube_file(user_parameters, user_parameters.slice_index_min, user_parameters.slice_index_max, PIX_X, PIX_Y, SQUARE_SIZE, false, true, false);
//		hypercube<T> Hypercube_file(user_parameters, user_parameters.slice_index_min, user_parameters.slice_index_max, 1024, 1024, SQUARE_SIZE, false, true, false);
		printf("Launching the ROHSA algorithm ...\n");
		algo_rohsa<T> algo(user_parameters, Hypercube_file);
	
		const std::string filename_raw = user_parameters.name_without_extension+".fits";
		std::cout<<"filename_raw = "<<filename_raw<<std::endl;	

		printf("Saving the result ...\n");
		Hypercube_file.save_grid_in_fits(user_parameters, algo.grid_params);


//		Hypercube_file.write_in_file(algo.grid_params, filename_raw);
	//	Hypercube_file.save_result(algo.grid_params, user_parameters);

	//	if(user_parameters.output_format_fits){
	//		Hypercube_file.save_grid_in_fits(user_parameters, algo.grid_params);
	//		printf("Result saved in fits file !\n");
	//	}else{
	//		Hypercube_file.save_result(algo.grid_params, user_parameters);
	//		printf("Result saved in dat file !\n");
	//	}

	if(user_parameters.save_noise_map_in_fits){
/*
			printf("Saving noise_map ...\n");
			Hypercube_file.save_noise_map_in_fits(user_parameters, algo.std_data);
*/
//			printf("Noise_map saved in fits file!\n");
		}
	}else{
	    hypercube<T> Hypercube_file(user_parameters, user_parameters.slice_index_min, user_parameters.slice_index_max); 
		printf("Launching the ROHSA algorithm ...\n");
		algo_rohsa<T> algo(user_parameters, Hypercube_file);

/*
		const std::string filename_raw = user_parameters.name_without_extension+".raw";
		std::cout<<"filename_raw = "<<filename_raw<<std::endl;	

		printf("Saving the result ...\n");
			//	Hypercube_file.write_in_file(algo.grid_params, filename_raw);
			//	Hypercube_file.save_result(algo.grid_params, user_parameters);
		Hypercube_file.save_grid_in_fits(user_parameters, algo.grid_params);

	//	if(user_parameters.output_format_fits){
	//		Hypercube_file.save_grid_in_fits(user_parameters, algo.grid_params);
	//		printf("Result saved in fits file !\n");
	//	}else{
	//		Hypercube_file.save_result(algo.grid_params, user_parameters);
	//		printf("Result saved in dat file !\n");
	//	}


		if(user_parameters.save_noise_map_in_fits){
			printf("Saving noise_map ...\n");
			Hypercube_file.save_noise_map_in_fits(user_parameters, algo.std_data);
			printf("Noise_map saved in fits file!\n");
		}
*/
	}
/*
	printf("Launching the ROHSA algorithm ...\n");
	algo_rohsa<T> algo(user_parameters, Hypercube_file);

	const std::string filename_raw = user_parameters.name_without_extension+".raw";
	std::cout<<"filename_raw = "<<filename_raw<<std::endl;	

	printf("Saving the result ...\n");
	Hypercube_file.write_in_file(algo.grid_params, filename_raw);
//	Hypercube_file.save_result(algo.grid_params, user_parameters);
	Hypercube_file.save_grid_in_fits(user_parameters, algo.grid_params);

//	if(user_parameters.output_format_fits){
//		Hypercube_file.save_grid_in_fits(user_parameters, algo.grid_params);
//		printf("Result saved in fits file !\n");
//	}else{
//		Hypercube_file.save_result(algo.grid_params, user_parameters);
//		printf("Result saved in dat file !\n");
//	}


	if(user_parameters.save_noise_map_in_fits){
		printf("Saving noise_map ...\n");
		Hypercube_file.save_noise_map_in_fits(user_parameters, algo.std_data);
		printf("Noise_map saved in fits file!\n");
	}
	printf("temps_test = %f\n");
*/
}

template void main_routine<double>(parameters<double>&);
template void main_routine<float>(parameters<float>&);

void test_min(char * argv[]){

	parameters<double> M;
	M.n_gauss = 12;
	M.lambda_amp = 10.;
	M.lambda_mu = 10.;
	M.lambda_sig = 10.;
	M.lambda_var_sig = 10.;
	int dim_x = 2;
	int dim_y = 2;
	int dim_v = 50;
	int n_beta = dim_x*dim_y*3*M.n_gauss+M.n_gauss;

	std::vector<std::vector<double>> std_map_vector(dim_y, std::vector<double>(dim_x,0.));
	std::vector<std::vector<std::vector<double>>> cube(dim_x, std::vector<std::vector<double>>(dim_y,std::vector<double>(dim_v,0.)));
	double* std_map_ = NULL;
	std_map_ = (double*)malloc(dim_x*dim_y*sizeof(double));
	double* beta_bis = NULL;
	beta_bis = (double*)malloc(n_beta*sizeof(double));
	double* lb_bis = NULL;
	lb_bis = (double*)malloc(n_beta*sizeof(double));
	double* ub_bis = NULL;
	ub_bis = (double*)malloc(n_beta*sizeof(double));
	double* cube_flattened = NULL;
	cube_flattened = (double*)malloc(dim_x*dim_y*dim_v*sizeof(double));

	if(true){
std_map_vector[        0][        0] =  0.290011077031847308571599342;
std_map_vector[        0][        1] =  0.240773854928421404686389451;
std_map_vector[        1][        0] =  0.238600479939023946140963517;
std_map_vector[        1][        1] =  0.207488736657283123765438404;
cube[        0][        0][        0] =  0.012778852668134277337230742;
cube[        1][        0][        0] = -0.031803278823417713283561170;
cube[        0][        1][        0] = -0.014872878013648005435243249;
cube[        1][        1][        0] = -0.052528199162679811706766486;
cube[        0][        0][        1] = -0.014009955965093467966653407;
cube[        1][        0][        1] = -0.033779682016756851226091385;
cube[        0][        1][        1] =  0.021095333067023602779954672;
cube[        1][        1][        1] = -0.030020716221997645334340632;
cube[        0][        0][        2] = -0.056338335810778517043218017;
cube[        1][        0][        2] = -0.022878963764242143952287734;
cube[        0][        1][        2] =  0.081864844343726872466504574;
cube[        1][        1][        2] = -0.025270139460644713835790753;
cube[        0][        0][        3] =  0.088717626164452667580917478;
cube[        1][        0][        3] = -0.074278819372466386994346976;
cube[        0][        1][        3] =  0.085196901311064721085131168;
cube[        1][        1][        3] =  0.001580067525537742767482996;
cube[        0][        0][        4] =  0.060092494743912538979202509;
cube[        1][        0][        4] = -0.028863122232905880082398653;
cube[        0][        1][        4] =  0.024304977021074591903015971;
cube[        1][        1][        4] =  0.006728563052092795260250568;
cube[        0][        0][        5] =  0.023110756711503199767321348;
cube[        1][        0][        5] = -0.037371180190348241012543440;
cube[        0][        1][        5] =  0.014867918345771613530814648;
cube[        1][        1][        5] =  0.009913395007515646284446120;
cube[        0][        0][        6] =  0.012818865864119288744404912;
cube[        1][        0][        6] =  0.020116851958846382331103086;
cube[        0][        1][        6] =  0.064604012084146233974024653;
cube[        1][        1][        6] =  0.016365571036203618859872222;
cube[        0][        0][        7] =  0.072962162633302796166390181;
cube[        1][        0][        7] =  0.079299675126549118431285024;
cube[        0][        1][        7] =  0.108328702548533328808844090;
cube[        1][        1][        7] =  0.014932929512724513188004494;
cube[        0][        0][        8] =  0.089158995157958997879177332;
cube[        1][        0][        8] =  0.138458627319778315722942352;
cube[        0][        1][        8] =  0.132756089812573918607085943;
cube[        1][        1][        8] =  0.056673499631870072335004807;
cube[        0][        0][        9] =  0.195370757456657884176820517;
cube[        1][        0][        9] =  0.186269001815162482671439648;
cube[        0][        1][        9] =  0.148177778479293920099735260;
cube[        1][        1][        9] =  0.115514951819022826384752989;
cube[        0][        0][       10] =  0.258068874434684403240680695;
cube[        1][        0][       10] =  0.202292169779866526369005442;
cube[        0][        1][       10] =  0.206274125257550622336566448;
cube[        1][        1][       10] =  0.167887601868642377667129040;
cube[        0][        0][       11] =  0.269419624210058827884495258;
cube[        1][        0][       11] =  0.254888567205853178165853024;
cube[        0][        1][       11] =  0.270166314630841952748596668;
cube[        1][        1][       11] =  0.231452832036666222847998142;
cube[        0][        0][       12] =  0.315952922162978211417794228;
cube[        1][        0][       12] =  0.394900985580534324981272221;
cube[        0][        1][       12] =  0.257220240750029915943741798;
cube[        1][        1][       12] =  0.251915058399390545673668385;
cube[        0][        0][       13] =  0.391976729197267559356987476;
cube[        1][        0][       13] =  0.435038198576876311562955379;
cube[        0][        1][       13] =  0.258807164151221513748168945;
cube[        1][        1][       13] =  0.273153571437433129176497459;
cube[        0][        0][       14] =  0.417686408698500599712133408;
cube[        1][        0][       14] =  0.417057976219439296983182430;
cube[        0][        1][       14] =  0.327180059990496374666690826;
cube[        1][        1][       14] =  0.359762601552574778907001019;
cube[        0][        0][       15] =  0.509740646273712627589702606;
cube[        1][        0][       15] =  0.406633443788450676947832108;
cube[        0][        1][       15] =  0.472174122205615276470780373;
cube[        1][        1][       15] =  0.390933892340399324893951416;
cube[        0][        0][       16] =  0.670352815828664461150765419;
cube[        1][        0][       16] =  0.466817472373804775997996330;
cube[        0][        1][       16] =  0.552935688105208100751042366;
cube[        1][        1][       16] =  0.364164935319422511383891106;
cube[        0][        0][       17] =  0.732137624872848391532897949;
cube[        1][        0][       17] =  0.578321816752577433362603188;
cube[        0][        1][       17] =  0.624483173189219087362289429;
cube[        1][        1][       17] =  0.460972146029234863817691803;
cube[        0][        0][       18] =  0.773960911075846524909138680;
cube[        1][        0][       18] =  0.582853887714009033516049385;
cube[        0][        1][       18] =  0.685127864202513592317700386;
cube[        1][        1][       18] =  0.564366915168648120015859604;
cube[        0][        0][       19] =  0.855874411638069432228803635;
cube[        1][        0][       19] =  0.620691371594148222357034683;
cube[        0][        1][       19] =  0.739878060827322769910097122;
cube[        1][        1][       19] =  0.580883781818556599318981171;
cube[        0][        0][       20] =  1.021955389289360027760267258;
cube[        1][        0][       20] =  0.726538494287524372339248657;
cube[        0][        1][       20] =  0.840863122211885638535022736;
cube[        1][        1][       20] =  0.675065997820638585835695267;
cube[        0][        0][       21] =  1.132568435401481110602617264;
cube[        1][        0][       21] =  0.887867100027506239712238312;
cube[        0][        1][       21] =  1.053786977878189645707607269;
cube[        1][        1][       21] =  0.774768042967480141669511795;
cube[        0][        0][       22] =  1.325747459704871289432048798;
cube[        1][        0][       22] =  0.963164429616881534457206726;
cube[        0][        1][       22] =  1.240942390046257060021162033;
cube[        1][        1][       22] =  0.853695798992703203111886978;
cube[        0][        0][       23] =  1.510921209992375224828720093;
cube[        1][        0][       23] =  1.119323435705155134201049805;
cube[        0][        1][       23] =  1.420078484145051334053277969;
cube[        1][        1][       23] =  1.009556996403262019157409668;
cube[        0][        0][       24] =  1.788338434926117770373821259;
cube[        1][        0][       24] =  1.355651928359293378889560699;
cube[        0][        1][       24] =  1.711827244143933057785034180;
cube[        1][        1][       24] =  1.288027571325073949992656708;
cube[        0][        0][       25] =  2.286279140491387806832790375;
cube[        1][        0][       25] =  1.722536755507462657988071442;
cube[        0][        1][       25] =  2.218440080920117907226085663;
cube[        1][        1][       25] =  1.720846681186230853199958801;
cube[        0][        0][       26] =  3.067261426331242546439170837;
cube[        1][        0][       26] =  2.364131384209031239151954651;
cube[        0][        1][       26] =  3.285828391293762251734733582;
cube[        1][        1][       26] =  2.677712213378981687128543854;
cube[        0][        0][       27] =  4.578669053385965526103973389;
cube[        1][        0][       27] =  3.513016085868002846837043762;
cube[        0][        1][       27] =  5.123928396089468151330947876;
cube[        1][        1][       27] =  4.404329656041227281093597412;
cube[        0][        0][       28] =  6.796058971260208636522293091;
cube[        1][        0][       28] =  5.817223934689536690711975098;
cube[        0][        1][       28] =  7.310504466411657631397247314;
cube[        1][        1][       28] =  6.977763366594444960355758667;
cube[        0][        0][       29] =  8.892955350980628281831741333;
cube[        1][        0][       29] =  8.929406175739131867885589600;
cube[        0][        1][       29] =  8.832945010042749345302581787;
cube[        1][        1][       29] = 10.312625287682749330997467041;
cube[        0][        0][       30] =  9.923850049264729022979736328;
cube[        1][        0][       30] = 11.239215968060307204723358154;
cube[        0][        1][       30] =  9.153606050065718591213226318;
cube[        1][        1][       30] = 12.770861712982878088951110840;
cube[        0][        0][       31] = 10.375904728134628385305404663;
cube[        1][        0][       31] = 11.029496484261471778154373169;
cube[        0][        1][       31] =  9.114428044704254716634750366;
cube[        1][        1][       31] = 12.449410675500985234975814819;
cube[        0][        0][       32] = 11.590244734310545027256011963;
cube[        1][        0][       32] =  9.524112521263305097818374634;
cube[        0][        1][       32] =  9.940333388862200081348419189;
cube[        1][        1][       32] = 10.763137411791831254959106445;
cube[        0][        0][       33] = 13.071582299075089395046234131;
cube[        1][        0][       33] =  9.594640898867510259151458740;
cube[        0][        1][       33] = 11.401192956429440528154373169;
cube[        1][        1][       33] = 10.334654536796733736991882324;
cube[        0][        0][       34] = 13.764754503732547163963317871;
cube[        1][        0][       34] = 11.877940978738479316234588623;
cube[        0][        1][       34] = 12.585399324423633515834808350;
cube[        1][        1][       34] = 11.087035635835491120815277100;
cube[        0][        0][       35] = 13.273808194557204842567443848;
cube[        1][        0][       35] = 13.963951316894963383674621582;
cube[        0][        1][       35] = 12.656154120806604623794555664;
cube[        1][        1][       35] = 11.479225183604285120964050293;
cube[        0][        0][       36] = 12.394814794883131980895996094;
cube[        1][        0][       36] = 14.374113938189111649990081787;
cube[        0][        1][       36] = 12.438034323044121265411376953;
cube[        1][        1][       36] = 11.582245592609979212284088135;
cube[        0][        0][       37] = 11.244784200796857476234436035;
cube[        1][        0][       37] = 13.353076888015493750572204590;
cube[        0][        1][       37] = 11.400840679300017654895782471;
cube[        1][        1][       37] = 10.701897033606655895709991455;
cube[        0][        0][       38] =  9.684088318492285907268524170;
cube[        1][        0][       38] = 11.462430277140811085700988770;
cube[        0][        1][       38] =  9.250333207077346742153167725;
cube[        1][        1][       38] =  8.815684594563208520412445068;
cube[        0][        0][       39] =  7.870528846746310591697692871;
cube[        1][        0][       39] =  9.218319732870440930128097534;
cube[        0][        1][       39] =  7.086549575906246900558471680;
cube[        1][        1][       39] =  6.813754790695384144783020020;
cube[        0][        0][       40] =  6.102857903635594993829727173;
cube[        1][        0][       40] =  7.209953077021054923534393311;
cube[        0][        1][       40] =  5.596067458624020218849182129;
cube[        1][        1][       40] =  5.374104208080098032951354980;
cube[        0][        0][       41] =  4.757920765201561152935028076;
cube[        1][        0][       41] =  5.762855147593654692173004150;
cube[        0][        1][       41] =  4.589042908279225230216979980;
cube[        1][        1][       41] =  4.504455547546967864036560059;
cube[        0][        0][       42] =  3.809546238975599408149719238;
cube[        1][        0][       42] =  4.769747985032154247164726257;
cube[        0][        1][       42] =  3.921226580976508557796478271;
cube[        1][        1][       42] =  4.040639605867909267544746399;
cube[        0][        0][       43] =  3.070203431969275698065757751;
cube[        1][        0][       43] =  3.973432027007220312952995300;
cube[        0][        1][       43] =  3.310543901956407353281974792;
cube[        1][        1][       43] =  3.630926545331021770834922791;
cube[        0][        0][       44] =  2.664169438678072765469551086;
cube[        1][        0][       44] =  3.331713892664993181824684143;
cube[        0][        1][       44] =  2.728852644097059965133666992;
cube[        1][        1][       44] =  3.155533730896422639489173889;
cube[        0][        0][       45] =  2.291968420729972422122955322;
cube[        1][        0][       45] =  2.661966283165384083986282349;
cube[        0][        1][       45] =  2.171358825697097927331924438;
cube[        1][        1][       45] =  2.527837066532811149954795837;
cube[        0][        0][       46] =  1.721187171977362595498561859;
cube[        1][        0][       46] =  2.117886983061907812952995300;
cube[        0][        1][       46] =  1.690181425714399665594100952;
cube[        1][        1][       46] =  1.991181566045270301401615143;
cube[        0][        0][       47] =  1.295058420742861926555633545;
cube[        1][        0][       47] =  1.700453797791851684451103210;
cube[        0][        1][       47] =  1.376722586705000139772891998;
cube[        1][        1][       47] =  1.526856320851948112249374390;
cube[        0][        0][       48] =  1.013319112913450226187705994;
cube[        1][        0][       48] =  1.368715326418168842792510986;
cube[        0][        1][       48] =  1.084158413665136322379112244;
cube[        1][        1][       48] =  1.229998264927417039871215820;
cube[        0][        0][       49] =  0.797654041787609457969665527;
cube[        1][        0][       49] =  1.231625161744887009263038635;
cube[        0][        1][       49] =  0.887737188269966281950473785;
cube[        1][        1][       49] =  1.016433492710348218679428101;
beta_bis[        0] =       5.2623872621991250042583;
beta_bis[        1] =      38.4244300760450343545926;
beta_bis[        2] =       2.3920299705868210971005;
beta_bis[        3] =       5.2612184067577496549006;
beta_bis[        4] =      36.8560292547541834551339;
beta_bis[        5] =       1.8046214235341380138777;
beta_bis[        6] =       2.5471345121489754603772;
beta_bis[        7] =      30.8977437725617853914173;
beta_bis[        8] =       1.3642776055762961817663;
beta_bis[        9] =       0.2042255789273528521210;
beta_bis[       10] =      13.1899304520676157181924;
beta_bis[       11] =       3.0898127109219926111905;
beta_bis[       12] =       4.5743980395251542248047;
beta_bis[       13] =      30.3263002086963560088861;
beta_bis[       14] =       2.0174232359443791118281;
beta_bis[       15] =       1.1319710898925230413425;
beta_bis[       16] =      34.9848906731428144212259;
beta_bis[       17] =       1.0000000000000000000000;
beta_bis[       18] =       0.7939160563169256334959;
beta_bis[       19] =      49.9399103114121416524540;
beta_bis[       20] =       9.9670525894346404527369;
beta_bis[       21] =       2.3661234274071758498792;
beta_bis[       22] =      43.2609399144657587044094;
beta_bis[       23] =       2.9203080886962240469984;
beta_bis[       24] =       1.6858713033446339757404;
beta_bis[       25] =      29.6726903683222751340054;
beta_bis[       26] =       7.0247629390301469243241;
beta_bis[       27] =       0.1075102424876830525813;
beta_bis[       28] =      17.7764117321557755246886;
beta_bis[       29] =       1.3929794455391244500930;
beta_bis[       30] =       5.3160356803871255948479;
beta_bis[       31] =      33.7186135049684381215229;
beta_bis[       32] =       1.9620655580132075890276;
beta_bis[       33] =       0.0000000000000000000000;
beta_bis[       34] =       4.0000008478891437846414;
beta_bis[       35] =       4.9999577941045725282265;
beta_bis[       36] =       5.2623872621991250042583;
beta_bis[       37] =      38.4244300760450343545926;
beta_bis[       38] =       2.3920299705868210971005;
beta_bis[       39] =       5.2612184067577496549006;
beta_bis[       40] =      36.8560292547541834551339;
beta_bis[       41] =       1.8046214235341380138777;
beta_bis[       42] =       2.5471345121489754603772;
beta_bis[       43] =      30.8977437725617853914173;
beta_bis[       44] =       1.3642776055762961817663;
beta_bis[       45] =       0.2042255789273528521210;
beta_bis[       46] =      13.1899304520676157181924;
beta_bis[       47] =       3.0898127109219926111905;
beta_bis[       48] =       4.5743980395251542248047;
beta_bis[       49] =      30.3263002086963560088861;
beta_bis[       50] =       2.0174232359443791118281;
beta_bis[       51] =       1.1319710898925230413425;
beta_bis[       52] =      34.9848906731428144212259;
beta_bis[       53] =       1.0000000000000000000000;
beta_bis[       54] =       0.7939160563169256334959;
beta_bis[       55] =      49.9399103114121416524540;
beta_bis[       56] =       9.9670525894346404527369;
beta_bis[       57] =       2.3661234274071758498792;
beta_bis[       58] =      43.2609399144657587044094;
beta_bis[       59] =       2.9203080886962240469984;
beta_bis[       60] =       1.6858713033446339757404;
beta_bis[       61] =      29.6726903683222751340054;
beta_bis[       62] =       7.0247629390301469243241;
beta_bis[       63] =       0.1075102424876830525813;
beta_bis[       64] =      17.7764117321557755246886;
beta_bis[       65] =       1.3929794455391244500930;
beta_bis[       66] =       5.3160356803871255948479;
beta_bis[       67] =      33.7186135049684381215229;
beta_bis[       68] =       1.9620655580132075890276;
beta_bis[       69] =       0.0000000000000000000000;
beta_bis[       70] =       4.0000008478891437846414;
beta_bis[       71] =       4.9999577941045725282265;
beta_bis[       72] =       5.2623872621991250042583;
beta_bis[       73] =      38.4244300760450343545926;
beta_bis[       74] =       2.3920299705868210971005;
beta_bis[       75] =       5.2612184067577496549006;
beta_bis[       76] =      36.8560292547541834551339;
beta_bis[       77] =       1.8046214235341380138777;
beta_bis[       78] =       2.5471345121489754603772;
beta_bis[       79] =      30.8977437725617853914173;
beta_bis[       80] =       1.3642776055762961817663;
beta_bis[       81] =       0.2042255789273528521210;
beta_bis[       82] =      13.1899304520676157181924;
beta_bis[       83] =       3.0898127109219926111905;
beta_bis[       84] =       4.5743980395251542248047;
beta_bis[       85] =      30.3263002086963560088861;
beta_bis[       86] =       2.0174232359443791118281;
beta_bis[       87] =       1.1319710898925230413425;
beta_bis[       88] =      34.9848906731428144212259;
beta_bis[       89] =       1.0000000000000000000000;
beta_bis[       90] =       0.7939160563169256334959;
beta_bis[       91] =      49.9399103114121416524540;
beta_bis[       92] =       9.9670525894346404527369;
beta_bis[       93] =       2.3661234274071758498792;
beta_bis[       94] =      43.2609399144657587044094;
beta_bis[       95] =       2.9203080886962240469984;
beta_bis[       96] =       1.6858713033446339757404;
beta_bis[       97] =      29.6726903683222751340054;
beta_bis[       98] =       7.0247629390301469243241;
beta_bis[       99] =       0.1075102424876830525813;
beta_bis[      100] =      17.7764117321557755246886;
beta_bis[      101] =       1.3929794455391244500930;
beta_bis[      102] =       5.3160356803871255948479;
beta_bis[      103] =      33.7186135049684381215229;
beta_bis[      104] =       1.9620655580132075890276;
beta_bis[      105] =       0.0000000000000000000000;
beta_bis[      106] =       4.0000008478891437846414;
beta_bis[      107] =       4.9999577941045725282265;
beta_bis[      108] =       5.2623872621991250042583;
beta_bis[      109] =      38.4244300760450343545926;
beta_bis[      110] =       2.3920299705868210971005;
beta_bis[      111] =       5.2612184067577496549006;
beta_bis[      112] =      36.8560292547541834551339;
beta_bis[      113] =       1.8046214235341380138777;
beta_bis[      114] =       2.5471345121489754603772;
beta_bis[      115] =      30.8977437725617853914173;
beta_bis[      116] =       1.3642776055762961817663;
beta_bis[      117] =       0.2042255789273528521210;
beta_bis[      118] =      13.1899304520676157181924;
beta_bis[      119] =       3.0898127109219926111905;
beta_bis[      120] =       4.5743980395251542248047;
beta_bis[      121] =      30.3263002086963560088861;
beta_bis[      122] =       2.0174232359443791118281;
beta_bis[      123] =       1.1319710898925230413425;
beta_bis[      124] =      34.9848906731428144212259;
beta_bis[      125] =       1.0000000000000000000000;
beta_bis[      126] =       0.7939160563169256334959;
beta_bis[      127] =      49.9399103114121416524540;
beta_bis[      128] =       9.9670525894346404527369;
beta_bis[      129] =       2.3661234274071758498792;
beta_bis[      130] =      43.2609399144657587044094;
beta_bis[      131] =       2.9203080886962240469984;
beta_bis[      132] =       1.6858713033446339757404;
beta_bis[      133] =      29.6726903683222751340054;
beta_bis[      134] =       7.0247629390301469243241;
beta_bis[      135] =       0.1075102424876830525813;
beta_bis[      136] =      17.7764117321557755246886;
beta_bis[      137] =       1.3929794455391244500930;
beta_bis[      138] =       5.3160356803871255948479;
beta_bis[      139] =      33.7186135049684381215229;
beta_bis[      140] =       1.9620655580132075890276;
beta_bis[      141] =       0.0000000000000000000000;
beta_bis[      142] =       4.0000008478891437846414;
beta_bis[      143] =       4.9999577941045725282265;
beta_bis[      144] =       2.3920299705868210971005;
beta_bis[      145] =       1.8046214235341380138777;
beta_bis[      146] =       1.3642776055762961817663;
beta_bis[      147] =       3.0898127109219926111905;
beta_bis[      148] =       2.0174232359443791118281;
beta_bis[      149] =       1.0000000000000000000000;
beta_bis[      150] =       9.9670525894346404527369;
beta_bis[      151] =       2.9203080886962240469984;
beta_bis[      152] =       7.0247629390301469243241;
beta_bis[      153] =       1.3929794455391244500930;
beta_bis[      154] =       1.9620655580132075890276;
beta_bis[      155] =       4.9999577941045725282265;
ub_bis[        0] =      13.7647545037325471639633;
ub_bis[        1] =      50.0000000000000000000000;
ub_bis[        2] =     100.0000000000000000000000;
ub_bis[        3] =      13.7647545037325471639633;
ub_bis[        4] =      50.0000000000000000000000;
ub_bis[        5] =     100.0000000000000000000000;
ub_bis[        6] =      13.7647545037325471639633;
ub_bis[        7] =      50.0000000000000000000000;
ub_bis[        8] =     100.0000000000000000000000;
ub_bis[        9] =      13.7647545037325471639633;
ub_bis[       10] =      50.0000000000000000000000;
ub_bis[       11] =     100.0000000000000000000000;
ub_bis[       12] =      13.7647545037325471639633;
ub_bis[       13] =      50.0000000000000000000000;
ub_bis[       14] =     100.0000000000000000000000;
ub_bis[       15] =      13.7647545037325471639633;
ub_bis[       16] =      50.0000000000000000000000;
ub_bis[       17] =     100.0000000000000000000000;
ub_bis[       18] =      13.7647545037325471639633;
ub_bis[       19] =      50.0000000000000000000000;
ub_bis[       20] =     100.0000000000000000000000;
ub_bis[       21] =      13.7647545037325471639633;
ub_bis[       22] =      50.0000000000000000000000;
ub_bis[       23] =     100.0000000000000000000000;
ub_bis[       24] =      13.7647545037325471639633;
ub_bis[       25] =      50.0000000000000000000000;
ub_bis[       26] =     100.0000000000000000000000;
ub_bis[       27] =      13.7647545037325471639633;
ub_bis[       28] =      50.0000000000000000000000;
ub_bis[       29] =     100.0000000000000000000000;
ub_bis[       30] =      13.7647545037325471639633;
ub_bis[       31] =      50.0000000000000000000000;
ub_bis[       32] =     100.0000000000000000000000;
ub_bis[       33] =      13.7647545037325471639633;
ub_bis[       34] =      50.0000000000000000000000;
ub_bis[       35] =     100.0000000000000000000000;
ub_bis[       36] =      12.6561541208066046237946;
ub_bis[       37] =      50.0000000000000000000000;
ub_bis[       38] =     100.0000000000000000000000;
ub_bis[       39] =      12.6561541208066046237946;
ub_bis[       40] =      50.0000000000000000000000;
ub_bis[       41] =     100.0000000000000000000000;
ub_bis[       42] =      12.6561541208066046237946;
ub_bis[       43] =      50.0000000000000000000000;
ub_bis[       44] =     100.0000000000000000000000;
ub_bis[       45] =      12.6561541208066046237946;
ub_bis[       46] =      50.0000000000000000000000;
ub_bis[       47] =     100.0000000000000000000000;
ub_bis[       48] =      12.6561541208066046237946;
ub_bis[       49] =      50.0000000000000000000000;
ub_bis[       50] =     100.0000000000000000000000;
ub_bis[       51] =      12.6561541208066046237946;
ub_bis[       52] =      50.0000000000000000000000;
ub_bis[       53] =     100.0000000000000000000000;
ub_bis[       54] =      12.6561541208066046237946;
ub_bis[       55] =      50.0000000000000000000000;
ub_bis[       56] =     100.0000000000000000000000;
ub_bis[       57] =      12.6561541208066046237946;
ub_bis[       58] =      50.0000000000000000000000;
ub_bis[       59] =     100.0000000000000000000000;
ub_bis[       60] =      12.6561541208066046237946;
ub_bis[       61] =      50.0000000000000000000000;
ub_bis[       62] =     100.0000000000000000000000;
ub_bis[       63] =      12.6561541208066046237946;
ub_bis[       64] =      50.0000000000000000000000;
ub_bis[       65] =     100.0000000000000000000000;
ub_bis[       66] =      12.6561541208066046237946;
ub_bis[       67] =      50.0000000000000000000000;
ub_bis[       68] =     100.0000000000000000000000;
ub_bis[       69] =      12.6561541208066046237946;
ub_bis[       70] =      50.0000000000000000000000;
ub_bis[       71] =     100.0000000000000000000000;
ub_bis[       72] =      14.3741139381891116499901;
ub_bis[       73] =      50.0000000000000000000000;
ub_bis[       74] =     100.0000000000000000000000;
ub_bis[       75] =      14.3741139381891116499901;
ub_bis[       76] =      50.0000000000000000000000;
ub_bis[       77] =     100.0000000000000000000000;
ub_bis[       78] =      14.3741139381891116499901;
ub_bis[       79] =      50.0000000000000000000000;
ub_bis[       80] =     100.0000000000000000000000;
ub_bis[       81] =      14.3741139381891116499901;
ub_bis[       82] =      50.0000000000000000000000;
ub_bis[       83] =     100.0000000000000000000000;
ub_bis[       84] =      14.3741139381891116499901;
ub_bis[       85] =      50.0000000000000000000000;
ub_bis[       86] =     100.0000000000000000000000;
ub_bis[       87] =      14.3741139381891116499901;
ub_bis[       88] =      50.0000000000000000000000;
ub_bis[       89] =     100.0000000000000000000000;
ub_bis[       90] =      14.3741139381891116499901;
ub_bis[       91] =      50.0000000000000000000000;
ub_bis[       92] =     100.0000000000000000000000;
ub_bis[       93] =      14.3741139381891116499901;
ub_bis[       94] =      50.0000000000000000000000;
ub_bis[       95] =     100.0000000000000000000000;
ub_bis[       96] =      14.3741139381891116499901;
ub_bis[       97] =      50.0000000000000000000000;
ub_bis[       98] =     100.0000000000000000000000;
ub_bis[       99] =      14.3741139381891116499901;
ub_bis[      100] =      50.0000000000000000000000;
ub_bis[      101] =     100.0000000000000000000000;
ub_bis[      102] =      14.3741139381891116499901;
ub_bis[      103] =      50.0000000000000000000000;
ub_bis[      104] =     100.0000000000000000000000;
ub_bis[      105] =      14.3741139381891116499901;
ub_bis[      106] =      50.0000000000000000000000;
ub_bis[      107] =     100.0000000000000000000000;
ub_bis[      108] =      12.7708617129828780889511;
ub_bis[      109] =      50.0000000000000000000000;
ub_bis[      110] =     100.0000000000000000000000;
ub_bis[      111] =      12.7708617129828780889511;
ub_bis[      112] =      50.0000000000000000000000;
ub_bis[      113] =     100.0000000000000000000000;
ub_bis[      114] =      12.7708617129828780889511;
ub_bis[      115] =      50.0000000000000000000000;
ub_bis[      116] =     100.0000000000000000000000;
ub_bis[      117] =      12.7708617129828780889511;
ub_bis[      118] =      50.0000000000000000000000;
ub_bis[      119] =     100.0000000000000000000000;
ub_bis[      120] =      12.7708617129828780889511;
ub_bis[      121] =      50.0000000000000000000000;
ub_bis[      122] =     100.0000000000000000000000;
ub_bis[      123] =      12.7708617129828780889511;
ub_bis[      124] =      50.0000000000000000000000;
ub_bis[      125] =     100.0000000000000000000000;
ub_bis[      126] =      12.7708617129828780889511;
ub_bis[      127] =      50.0000000000000000000000;
ub_bis[      128] =     100.0000000000000000000000;
ub_bis[      129] =      12.7708617129828780889511;
ub_bis[      130] =      50.0000000000000000000000;
ub_bis[      131] =     100.0000000000000000000000;
ub_bis[      132] =      12.7708617129828780889511;
ub_bis[      133] =      50.0000000000000000000000;
ub_bis[      134] =     100.0000000000000000000000;
ub_bis[      135] =      12.7708617129828780889511;
ub_bis[      136] =      50.0000000000000000000000;
ub_bis[      137] =     100.0000000000000000000000;
ub_bis[      138] =      12.7708617129828780889511;
ub_bis[      139] =      50.0000000000000000000000;
ub_bis[      140] =     100.0000000000000000000000;
ub_bis[      141] =      12.7708617129828780889511;
ub_bis[      142] =      50.0000000000000000000000;
ub_bis[      143] =     100.0000000000000000000000;
ub_bis[      144] =     100.0000000000000000000000;
ub_bis[      145] =     100.0000000000000000000000;
ub_bis[      146] =     100.0000000000000000000000;
ub_bis[      147] =     100.0000000000000000000000;
ub_bis[      148] =     100.0000000000000000000000;
ub_bis[      149] =     100.0000000000000000000000;
ub_bis[      150] =     100.0000000000000000000000;
ub_bis[      151] =     100.0000000000000000000000;
ub_bis[      152] =     100.0000000000000000000000;
ub_bis[      153] =     100.0000000000000000000000;
ub_bis[      154] =     100.0000000000000000000000;
ub_bis[      155] =     100.0000000000000000000000;
lb_bis[        0] =       0.0000000000000000000000;
lb_bis[        1] =       0.0000000000000000000000;
lb_bis[        2] =       1.0000000000000000000000;
lb_bis[        3] =       0.0000000000000000000000;
lb_bis[        4] =       0.0000000000000000000000;
lb_bis[        5] =       1.0000000000000000000000;
lb_bis[        6] =       0.0000000000000000000000;
lb_bis[        7] =       0.0000000000000000000000;
lb_bis[        8] =       1.0000000000000000000000;
lb_bis[        9] =       0.0000000000000000000000;
lb_bis[       10] =       0.0000000000000000000000;
lb_bis[       11] =       1.0000000000000000000000;
lb_bis[       12] =       0.0000000000000000000000;
lb_bis[       13] =       0.0000000000000000000000;
lb_bis[       14] =       1.0000000000000000000000;
lb_bis[       15] =       0.0000000000000000000000;
lb_bis[       16] =       0.0000000000000000000000;
lb_bis[       17] =       1.0000000000000000000000;
lb_bis[       18] =       0.0000000000000000000000;
lb_bis[       19] =       0.0000000000000000000000;
lb_bis[       20] =       1.0000000000000000000000;
lb_bis[       21] =       0.0000000000000000000000;
lb_bis[       22] =       0.0000000000000000000000;
lb_bis[       23] =       1.0000000000000000000000;
lb_bis[       24] =       0.0000000000000000000000;
lb_bis[       25] =       0.0000000000000000000000;
lb_bis[       26] =       1.0000000000000000000000;
lb_bis[       27] =       0.0000000000000000000000;
lb_bis[       28] =       0.0000000000000000000000;
lb_bis[       29] =       1.0000000000000000000000;
lb_bis[       30] =       0.0000000000000000000000;
lb_bis[       31] =       0.0000000000000000000000;
lb_bis[       32] =       1.0000000000000000000000;
lb_bis[       33] =       0.0000000000000000000000;
lb_bis[       34] =       0.0000000000000000000000;
lb_bis[       35] =       1.0000000000000000000000;
lb_bis[       36] =       0.0000000000000000000000;
lb_bis[       37] =       0.0000000000000000000000;
lb_bis[       38] =       1.0000000000000000000000;
lb_bis[       39] =       0.0000000000000000000000;
lb_bis[       40] =       0.0000000000000000000000;
lb_bis[       41] =       1.0000000000000000000000;
lb_bis[       42] =       0.0000000000000000000000;
lb_bis[       43] =       0.0000000000000000000000;
lb_bis[       44] =       1.0000000000000000000000;
lb_bis[       45] =       0.0000000000000000000000;
lb_bis[       46] =       0.0000000000000000000000;
lb_bis[       47] =       1.0000000000000000000000;
lb_bis[       48] =       0.0000000000000000000000;
lb_bis[       49] =       0.0000000000000000000000;
lb_bis[       50] =       1.0000000000000000000000;
lb_bis[       51] =       0.0000000000000000000000;
lb_bis[       52] =       0.0000000000000000000000;
lb_bis[       53] =       1.0000000000000000000000;
lb_bis[       54] =       0.0000000000000000000000;
lb_bis[       55] =       0.0000000000000000000000;
lb_bis[       56] =       1.0000000000000000000000;
lb_bis[       57] =       0.0000000000000000000000;
lb_bis[       58] =       0.0000000000000000000000;
lb_bis[       59] =       1.0000000000000000000000;
lb_bis[       60] =       0.0000000000000000000000;
lb_bis[       61] =       0.0000000000000000000000;
lb_bis[       62] =       1.0000000000000000000000;
lb_bis[       63] =       0.0000000000000000000000;
lb_bis[       64] =       0.0000000000000000000000;
lb_bis[       65] =       1.0000000000000000000000;
lb_bis[       66] =       0.0000000000000000000000;
lb_bis[       67] =       0.0000000000000000000000;
lb_bis[       68] =       1.0000000000000000000000;
lb_bis[       69] =       0.0000000000000000000000;
lb_bis[       70] =       0.0000000000000000000000;
lb_bis[       71] =       1.0000000000000000000000;
lb_bis[       72] =       0.0000000000000000000000;
lb_bis[       73] =       0.0000000000000000000000;
lb_bis[       74] =       1.0000000000000000000000;
lb_bis[       75] =       0.0000000000000000000000;
lb_bis[       76] =       0.0000000000000000000000;
lb_bis[       77] =       1.0000000000000000000000;
lb_bis[       78] =       0.0000000000000000000000;
lb_bis[       79] =       0.0000000000000000000000;
lb_bis[       80] =       1.0000000000000000000000;
lb_bis[       81] =       0.0000000000000000000000;
lb_bis[       82] =       0.0000000000000000000000;
lb_bis[       83] =       1.0000000000000000000000;
lb_bis[       84] =       0.0000000000000000000000;
lb_bis[       85] =       0.0000000000000000000000;
lb_bis[       86] =       1.0000000000000000000000;
lb_bis[       87] =       0.0000000000000000000000;
lb_bis[       88] =       0.0000000000000000000000;
lb_bis[       89] =       1.0000000000000000000000;
lb_bis[       90] =       0.0000000000000000000000;
lb_bis[       91] =       0.0000000000000000000000;
lb_bis[       92] =       1.0000000000000000000000;
lb_bis[       93] =       0.0000000000000000000000;
lb_bis[       94] =       0.0000000000000000000000;
lb_bis[       95] =       1.0000000000000000000000;
lb_bis[       96] =       0.0000000000000000000000;
lb_bis[       97] =       0.0000000000000000000000;
lb_bis[       98] =       1.0000000000000000000000;
lb_bis[       99] =       0.0000000000000000000000;
lb_bis[      100] =       0.0000000000000000000000;
lb_bis[      101] =       1.0000000000000000000000;
lb_bis[      102] =       0.0000000000000000000000;
lb_bis[      103] =       0.0000000000000000000000;
lb_bis[      104] =       1.0000000000000000000000;
lb_bis[      105] =       0.0000000000000000000000;
lb_bis[      106] =       0.0000000000000000000000;
lb_bis[      107] =       1.0000000000000000000000;
lb_bis[      108] =       0.0000000000000000000000;
lb_bis[      109] =       0.0000000000000000000000;
lb_bis[      110] =       1.0000000000000000000000;
lb_bis[      111] =       0.0000000000000000000000;
lb_bis[      112] =       0.0000000000000000000000;
lb_bis[      113] =       1.0000000000000000000000;
lb_bis[      114] =       0.0000000000000000000000;
lb_bis[      115] =       0.0000000000000000000000;
lb_bis[      116] =       1.0000000000000000000000;
lb_bis[      117] =       0.0000000000000000000000;
lb_bis[      118] =       0.0000000000000000000000;
lb_bis[      119] =       1.0000000000000000000000;
lb_bis[      120] =       0.0000000000000000000000;
lb_bis[      121] =       0.0000000000000000000000;
lb_bis[      122] =       1.0000000000000000000000;
lb_bis[      123] =       0.0000000000000000000000;
lb_bis[      124] =       0.0000000000000000000000;
lb_bis[      125] =       1.0000000000000000000000;
lb_bis[      126] =       0.0000000000000000000000;
lb_bis[      127] =       0.0000000000000000000000;
lb_bis[      128] =       1.0000000000000000000000;
lb_bis[      129] =       0.0000000000000000000000;
lb_bis[      130] =       0.0000000000000000000000;
lb_bis[      131] =       1.0000000000000000000000;
lb_bis[      132] =       0.0000000000000000000000;
lb_bis[      133] =       0.0000000000000000000000;
lb_bis[      134] =       1.0000000000000000000000;
lb_bis[      135] =       0.0000000000000000000000;
lb_bis[      136] =       0.0000000000000000000000;
lb_bis[      137] =       1.0000000000000000000000;
lb_bis[      138] =       0.0000000000000000000000;
lb_bis[      139] =       0.0000000000000000000000;
lb_bis[      140] =       1.0000000000000000000000;
lb_bis[      141] =       0.0000000000000000000000;
lb_bis[      142] =       0.0000000000000000000000;
lb_bis[      143] =       1.0000000000000000000000;
lb_bis[      144] =       1.0000000000000000000000;
lb_bis[      145] =       1.0000000000000000000000;
lb_bis[      146] =       1.0000000000000000000000;
lb_bis[      147] =       1.0000000000000000000000;
lb_bis[      148] =       1.0000000000000000000000;
lb_bis[      149] =       1.0000000000000000000000;
lb_bis[      150] =       1.0000000000000000000000;
lb_bis[      151] =       1.0000000000000000000000;
lb_bis[      152] =       1.0000000000000000000000;
lb_bis[      153] =       1.0000000000000000000000;
lb_bis[      154] =       1.0000000000000000000000;
lb_bis[      155] =       1.0000000000000000000000;

	}

	for(int i=0; i<dim_y; i++){
		for(int j=0; j<dim_x; j++){
			std_map_[i*dim_x+j]=std_map_vector[i][j];
		}
	}

			for(int k=0; k<dim_v; k++){
		for(int i=0; i<dim_y; i++){
	for(int j=0; j<dim_x; j++){
			cube_flattened[k*dim_x*dim_y+i*dim_x+j]=cube[j][i][k];
			}
		}
	}

	minimize_fortran_test(M, long(n_beta), long(10), beta_bis, lb_bis, ub_bis, cube, std_map_, dim_x, dim_y, dim_v, cube_flattened);

	for(int i=0; i<n_beta; i++){
		printf("beta_cpp[%d] = %.26f;\n",i,beta_bis[i]);
	}

	free(ub_bis);
	free(lb_bis);
	free(cube_flattened);
	free(beta_bis);
	free(std_map_);
}

int main(int argc, char * argv[])
{

//	minimize_float();
//	minimize();
//	exit(0);
//	test_min(argv);cmake ../ && make &&
//	exit(0);

/*
//(double* f, double* g, double* cube, double* beta, int dim_v, int dim_y, int dim_x, int n_gauss, double* kernel, double lambda_amp, double lambda_mu, double lambda_sig, double lambda_var_amp, double lambda_var_mu, double lambda_var_sig, double* std_map);
	double beta[]= {1.,2.,3., 4., 5., 6., 7., 8., 9.};
	double f = 0.;
	double lambda_one = 1.;
	double lambda_zero = 0.;
	double g[]= {0.,0.,0.};
	double cube[]= {1.,1.,1.};
	double std_map[]= {1.};
	int dim_x = 1;
	int dim_y = 1;
	int dim_v = 3;
	int n_gauss = 1;
	double x[3];
	x[0] = 1.;
	x[1] = 1.;
	x[2] = 1.;
	double kernel[]= {0.,-0.25,0.,-0.25,1,-0.25,0.,-0.25,0.};
	printf("TEST !\n");
//	print_a_real_(&x[0]);

	int length_beta = 3;
	int length_cube_0 = 1;
	int length_cube_1 = 1;
	int length_cube_2 = 3;
	int length_kernel_0 = 3;
	int length_kernel_1 = 3;
	int length_std_map_0 = 1;
	int length_std_map_1 = 1;
	int length_g = 3;

	print_vec_(beta, &dim_v,&dim_v);

	f_g_cube_fast_(&f, g, cube, x, &dim_v, &dim_x, &dim_x, &n_gauss, kernel, 
&lambda_one, &lambda_one, &lambda_one, &lambda_zero, &lambda_zero, 
&lambda_one, std_map, &length_beta, &length_cube_0, &length_cube_1, 
&length_cube_2, &length_kernel_0, &length_kernel_1, &length_std_map_0, 
&length_std_map_1, &length_g);

//	array_(&dim_v,x);
	printf("x[0] = %f\n", x[0]);
	printf("x[1] = %f\n", x[1]);
	printf("x[2] = %f\n", x[2]);


////	test_ter_same_order(argv);
	exit(0);
	*/
	std::cout<<"argv = "<<argv[1]<<" , "<<argv[2]<<" , "<<argv[3]<<" , "<<argv[4]<<std::endl;
	parameters<float> user_parametres_float(argv[1], argv[2], argv[3], argv[4]);
	parameters<double> user_parametres_double(argv[1], argv[2], argv[3], argv[4]);

	if(user_parametres_double.double_mode){
		main_routine<double>(user_parametres_double);
	}else if(user_parametres_float.float_mode){
		main_routine<float>(user_parametres_float);
	}

	return 0;
}

