#include "algo_rohsa.h"
#include <omp.h>


algo_rohsa::algo_rohsa(model &M, const hypercube &Hypercube)
{
	this->file = Hypercube; //The hypercube is not modified then
	this->dim_data = Hypercube.get_dim_data();
	this->dim_x = dim_data[2];
	this->dim_y = dim_data[1];
	this->dim_v = dim_data[0];

	double temps1_descente = omp_get_wtime();	
	std_spectrum(dim_x, dim_y, dim_v); //oublier

	mean_spectrum(dim_x, dim_y, dim_v);
	max_spectrum(dim_x, dim_y, dim_v); //oublier

	std::vector<double> b_params(M.n_gauss,0.);

	double max_mean_spect = *std::max_element(mean_spect.begin(), mean_spect.end());

	max_spectrum_norm(dim_x,dim_y, dim_v, max_mean_spect);

	std::cout << " descent : "<< M.descent << std::endl;


	if(M.descent)
	{		
//		std::vector<std::vector<std::vector<double>>> grid_params(3*(M.n_gauss+(file.nside*M.n_gauss_add)), std::vector<std::vector<double>>(dim_y, std::vector<double>(dim_x)));
//		std::vector<std::vector<std::vector<double>>> fit_params(3*(M.n_gauss+(file.nside*M.n_gauss_add)), std::vector<std::vector<double>>(1, std::vector<double>(1,1.)));

		std::vector<std::vector<double>> sheet(dim_y,std::vector<double>(dim_x));
		std::vector<std::vector<double>> sheet_(1,std::vector<double>(1,0.));

		std::cout<<"nside = "<< file.nside<<std::endl;
		
		for(int compteur(0) ; compteur< 3*(M.n_gauss+(file.nside*M.n_gauss_add)); compteur++)
		{
			M.grid_params.vector::push_back(sheet);
		}
		for(int compteur(0); compteur<3*(M.n_gauss+(file.nside*M.n_gauss_add)); compteur++) {
			M.fit_params.vector::push_back(sheet_);
		}
		
		sheet.clear();
		sheet_.clear();
		
//		std::cout << "test fit_params : "<<fit_params[0][0][0]<<std::endl;
		
		for(int i(0);i<M.n_gauss; i++){
			std::cout<<"TEST"<<std::endl;
			M.fit_params[0+3*i][0][0] = 0.;
			M.fit_params[1+3*i][0][0] = 1.;
			M.fit_params[2+3*i][0][0] = 1.;
		}
		for(int i(0); i<M.fit_params.size();i++){
			std::cout<<"fit_params["<<i<<"][0][0] = "<<M.fit_params[i][0][0]<<std::endl;
		}

	} else {
//		std::vector<std::vector<std::vector<double>>> fit_params(3*(M.n_gauss+n_gauss_add), std::vector<std::vector<double>>(dim_y, std::vector<double>(dim_x)));
				
		std::vector<std::vector<double>> sheet(dim_y,std::vector<double>(dim_x));
		
		for(int compteur(0); compteur<3*(M.n_gauss+n_gauss_add) ; compteur++)
		{
			M.grid_params.vector::push_back(sheet);
		}

		sheet.clear();

	}


	std::cout << "fit_params.size() : " << M.fit_params.size() << " , " << M.fit_params[0].size() << " , " << M.fit_params[0][0].size() <<  std::endl;
	std::cout << "grid_params.size() : "<< M.grid_params.size() << " , " << M.grid_params[0].size()  << " , " << M.grid_params[0][0].size() << std::endl;

	std::vector<double> fit_params_flat(M.fit_params.size(),0.); //used below

	if(M.descent)
	{
		double temps_upgrade=0.;
		double temps_multiresol=0.;
		double temps_init = 0.;
		double temps_mean_array=0.;
		int n;
//		#pragma omp parallel private(n) shared(temps_upgrade, temps_multiresol, temps_init, temps_mean_array,M, fit_params_flat,file)
//		{
//		#pragma omp for 
		for(n=0; n<file.nside; n++)
		{
			int power(pow(2,n));
			double temps1_multiresol = omp_get_wtime();
			file.multiresolution(n+1);
			double temps2_multiresol = omp_get_wtime();
			temps_multiresol += temps2_multiresol-temps1_multiresol;

			std::cout << "fit_params.size() : " << M.fit_params.size() << " , " << M.fit_params[0].size() << " , " << M.fit_params[0][0].size() <<  std::endl;
	std::cout << "grid_params.size() : "<< M.grid_params.size() << " , " << M.grid_params[0].size()  << " , " << M.grid_params[0][0].size() << std::endl;

			std::cout << " power = " << power << std::endl;

			std::vector<std::vector<std::vector<double>>> cube_mean(power, std::vector<std::vector<double>>(power,std::vector<double>(dim_v,1.)));

			double temps1_mean_array = omp_get_wtime();
			mean_array(power, cube_mean);
			double temps2_mean_array = omp_get_wtime();
			temps_mean_array+=temps2_mean_array-temps1_mean_array;

			std::vector<double> cube_mean_flat(cube_mean[0][0].size());
			
			for(int e(0); e<M.fit_params.size(); e++) {
				fit_params_flat[e] = M.fit_params[0][0][e];
				std::cout << "TEST valeurs e ==> " << e<<  std::endl;
			}
			for(int e(0); e<cube_mean[0][0].size(); e++) {
				cube_mean_flat[e] = cube_mean[0][0][e]; //cache
			}			
			if (n==0) {
//				std::cout<<"////taille fit_params_flat : "<<fit_params_flat.size()<<std::endl;
				//assume option "mean"
				std::cout<<"Init mean spectrum"<<std::endl;
				double temps1_init = omp_get_wtime();
				init_spectrum(M, cube_mean_flat, fit_params_flat);
				for(int e(0); e<fit_params.size(); e++){
				std::cout<<"fit_params_flat["<<e<<"] = "<<fit_params_flat[e]<<std::endl;
				}	
				double temps2_init = omp_get_wtime();
				temps_init += temps2_init - temps1_init;
			for(int i(0); i<M.n_gauss; i++) {
				b_params[i]= fit_params_flat[2+3*i];
			}
			}
			if (true) {//regul == false
				for(int e(0); e<M.fit_params.size(); e++) {
					M.fit_params[0][0][e]=fit_params_flat[e];
					M.grid_params[0][0][e] = M.fit_params[0][0][e];
					}
				double temps1_upgrade = omp_get_wtime();
				upgrade(M ,cube_mean, M.grid_params, power);
				double temps2_upgrade = omp_get_wtime();
				temps_upgrade+=temps2_upgrade-temps1_upgrade;
			}
			
//			go_up_level(M.grid_params);
//		}
		}
		double temps2_descente = omp_get_wtime();
		
		std::cout<<"Temps TOTAL de descente : "<<temps2_descente - temps1_descente <<std::endl;
		std::cout<<"Temps de upgrade : "<< temps_upgrade <<std::endl;
		std::cout<<"Temps de multirésolution : "<< temps_multiresol <<std::endl;
		std::cout<<"Temps de mean_array : "<<temps_mean_array<<std::endl;
		std::cout<<"Temps de init : "<<temps_init<<std::endl;
	}
}

void algo_rohsa::upgrade(model &M, std::vector<std::vector<std::vector<double>>> &cube, std::vector<std::vector<std::vector<double>>> params, int power) {
	std::vector<double> line(dim_v,0.);
	std::vector<double> x(3*M.n_gauss,0.); 
	std::vector<double> lb(3*M.n_gauss,0.);
	std::vector<double> ub(3*M.n_gauss,0.);
	int i,j;

	#pragma omp parallel num_threads(2) shared(cube,params) shared(power) shared(M)
	{
	printf("thread:%d\n", omp_get_thread_num());
	#pragma omp for private(i,j,lb,ub,x,line)
	for(i=0;i<power; i++){
		for(j=0;j<power; j++){
			std::cout << "0 ==> TEST"<<  std::endl;
			int p;
			for(p=0; p<cube[0][0].size();p++){
				line[p]=cube[i][j][p];
			}

			for(p=0; p<params.size(); p++){
				x[p]=params[p][i][j]; //cache
			}
			init_bounds(M, line, M.n_gauss, lb, ub);
			minimize_spec(M,3*M.n_gauss ,M.m ,x ,lb ,ub ,line);
			for(p=0; p<params.size();p++){
				params[p][i][j]=x[p]; //cache
//				std::cout << "p = "<<p<<  std::endl;
			}
		}
	}
	}
}
/*
void algo_rohsa::go_up_level(std::vector<std::vector<std::vector<double>>> &cube_params) {
	double dimension = cube_params.size();
	std::vector<std::vector<std::vector<double>>> cube_params_down = cube_params;	
}
*/

void algo_rohsa::init_bounds(model &M, std::vector<double> line, int n_gauss_local, std::vector<double> lb, std::vector<double> ub) {

	double max_line = *std::max_element(line.begin(), line.end());

	for(int i(0); i<n_gauss_local; i++) {

		lb[0+3*i]=0.;
		ub[0+3*i]=max_line;

		lb[1+3*i]=0.;
		ub[1+3*i]=dim_v;


		lb[2+3*i]=M.lb_sig;
		ub[2+3*i]=M.ub_sig;
	}
}

double algo_rohsa::model_function(int x, double a, double m, double s) {

	return a*exp(-pow((double(x)-m),2.) / (2.*pow(s,2.)));

}

int algo_rohsa::minloc(std::vector<double> tab) {
	return std::distance(tab.begin(), std::min_element( tab.begin()+1, tab.end() ));
}

void algo_rohsa::minimize_spec(model &M, long n, long m, std::vector<double> x_v, std::vector<double> lb_v, std::vector<double> ub_v, std::vector<double> line_v) {
/* Minimize_spec */ 
//int MAIN__(void)
    std::cout << "3 ==> TEST dim_x = "<< x_v.size() <<std::endl;
    std::vector<double> _residual_;
    for(int p(0); p<dim_v; p++){
	_residual_.vector::push_back(0.);
    }
    std::cout << "3 ==> TEST dim_residual = " << _residual_.size()<< std::endl;

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
    int line[line_v.size()];
    
    for(int i(0); i<x_v.size(); i++) {
	x[i]=x_v[i];
	std::cout<<" x["<<i<<"]= "<<x[i]<<std::endl;
    } 
    for(int i(0); i<lb_v.size(); i++) {
	lb[i]=lb_v[i];
    } 
    for(int i(0); i<ub_v.size(); i++) {
	ub[i]=ub_v[i];
    } 
    for(int i(0); i<line_v.size(); i++) {
	line[i]=line_v[i];
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
    /*     This is the call to the L-BFGS-B code. */
    std::cout<<" Début appel BFGS "<<std::endl;

/*
    for(int p(0); p<9; p++){
        std::cout<<" x["<<p<<"]= "<<x[p]<<std::endl;
    }
*/
//    std::cout<<" x["<<p<<"]= "<<x[p]<<std::endl;

//	std::cout << "TEST 1 /!\ fit_params.size() : " << M.fit_params.size() << " , " << M.fit_params[0].size() << " , " << M.fit_params[0][0].size() <<  std::endl;
    setulb(&n, &m, x, lb, ub, nbd, &f, g, &factr, &pgtol, wa, iwa, task, 
            &M.iprint, csave, lsave, isave, dsave);

/*     if (s_cmp(task, "FG", (ftnlen)2, (ftnlen)2) == 0) { */
    if ( IS_FG(*task) ) {
	myresidual(M, x, line, _residual_);
	f = myfunc_spec(_residual_);
	mygrad_spec(M, g, _residual_, x);
//	std::cout << "TEST 2 /!\ fit_params.size() : " << M.fit_params.size() << " , " << M.fit_params[0].size() << " , " << M.fit_params[0][0].size() <<  std::endl;
    if (*task==NEW_X ) {
	if (isave[33] >= M.maxiter) {
		*task = STOP_ITER;
		}
	
	if (dsave[12] <= (fabs(f) + 1.) * 1e-10) {
		*task = STOP_GRAD;
	}
     }

        /*          go back to the minimization routine. */
        goto L111;
    }

}

void algo_rohsa::old_minimize_spec(model &M, long n, long m, std::vector<double> x_v, std::vector<double> lb_v, std::vector<double> ub_v, std::vector<double> line_v) {
/* Minimize_spec */ 
//int MAIN__(void)
    std::cout << "2,5/////////TEST dim_v = "<< dim_v <<std::endl;
    std::vector<double> _residual_(dim_v, 0.);
    std::cout << "2,5/////////TEST" << std::endl;

    /* System generated locals */
    long i__1;
    double d__1, d__2;
    /* Local variables */
    double f, g[n];
    static long i__;

    int taille_wa = 2*m*n+5*n+11*m*m+8*m;
    int taille_iwa = 3*n;
    double t1, t2, wa[taille_wa];
    long nbd[n], iwa[taille_iwa];
/*     static char task[60]; */
    static long taskValue;
    static long *task=&taskValue; /* must initialize !! */
/*      http://stackoverflow.com/a/11278093/269192 */
    static double factr;
/*     static char csave[60]; */
    static long csaveValue;
    static long *csave=&csaveValue;
    static double dsave[29];
    static long isave[44];
    static logical lsave[4];
    static double pgtol;
//    static long iprint;

// converts the vectors into a regular list
    double x[x_v.size()];
    double lb[lb_v.size()];
    double ub[ub_v.size()];
    int line[line_v.size()];
    
    for(int i(0); i<x_v.size(); i++) {
	x[i]=x_v[i];
    } 
    for(int i(0); i<lb_v.size(); i++) {
	lb[i]=lb_v[i];
    } 
    for(int i(0); i<ub_v.size(); i++) {
	ub[i]=ub_v[i];
    } 
    for(int i(0); i<line_v.size(); i++) {
	line[i]=line_v[i];
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
    /*     This is the call to the L-BFGS-B code. */
//	std::cout << "TEST 1 /!\ fit_params.size() : " << M.fit_params.size() << " , " << M.fit_params[0].size() << " , " << M.fit_params[0][0].size() <<  std::endl;
    setulb(&n, &m, x, lb, ub, nbd, &f, g, &factr, &pgtol, wa, iwa, task, 
            &M.iprint, csave, lsave, isave, dsave);

/*     if (s_cmp(task, "FG", (ftnlen)2, (ftnlen)2) == 0) { */
    if ( IS_FG(*task) ) {
	myresidual(M, x, line, _residual_);
	f = myfunc_spec(_residual_);
	mygrad_spec(M, g, _residual_, x);
//	std::cout << "TEST 2 /!\ fit_params.size() : " << M.fit_params.size() << " , " << M.fit_params[0].size() << " , " << M.fit_params[0][0].size() <<  std::endl;
    if (*task==NEW_X ) {
	if (isave[33] >= M.maxiter) {
		*task = STOP_ITER;
		}
	
	if (dsave[12] <= (fabs(f) + 1.) * 1e-10) {
		*task = STOP_GRAD;
	}
     }

        /*          go back to the minimization routine. */
        goto L111;
    }

}

double algo_rohsa::myfunc_spec(std::vector<double> &residual) {
	double S(0.);
	for(int p(0); p<residual.size(); p++) {
		S+=pow(residual[p],2);
	}
	return 0.5*S;
}

void algo_rohsa::myresidual(model &M, double params[], int line[], std::vector<double> residual) {
	double g(0.);	
	int k;
	std::vector<double> model(dim_v,0.);
	for(int i(0); i<M.n_gauss; i++) {
		for(k=1; k<=dim_v; k++) {
			model[k-1]+= model_function(k, params[3*i], params[1+3*i], params[2+3*i]);
		}
	}

	for(int p=0; p<residual.size(); p++) {
		residual[p]=model[p]-line[p]; 
	}
}

void algo_rohsa::mygrad_spec(model &M, double gradient[], std::vector<double> &residual, double params[]) {
	std::vector<std::vector<double>> dF_over_dB(3*M.n_gauss, std::vector<double>(dim_v,0.));
	double g(0.);
	for(int p(0); p<int( sizeof(gradient)/sizeof(gradient[0]) ); p++) {
		gradient[p]=0.;
	}
	for(int i(0); i<M.n_gauss; i++) {
		for(int k(0); k<dim_v; k++) {
			dF_over_dB[0+3*i][k] += exp(-( double(k)-pow(params[1+3*i],2.) )/2*pow(params[2+3*i],.2) );
			dF_over_dB[1+3*i][k] += params[3*i]*( double(k) - params[1+3*i])/(pow(params[2+3*i],2.)) * exp(-( double(k)-pow(params[1+3*i],2.) )/(2*pow(params[2+3*i],.2)) );
			dF_over_dB[2+3*i][k] += params[3*i]*pow( double(k) - params[1+3*i] , 2.)/(pow(params[2+3*i],3.)) * exp(-( double(k)-pow(params[1+3*i],2.) )/(2*pow(params[2+3*i],.2)) );
		}
	}
	for(int i(0); i<dim_v; i++){
		for(int k(0); k<M.n_gauss; k++){
			gradient[k]+=dF_over_dB[k][i]*residual[i];
		}
	}
}

void algo_rohsa::init_spectrum(model &M, std::vector<double> line, std::vector<double> params) {
	std::vector<double> model_tab(dim_v,0.);
	std::vector<double> residual(dim_v,0.);
//	std::cout<<"////taille params : "<<params.size()<<std::endl;
	for(int i(1); i<=M.n_gauss; i++) {
		std::vector<double> lb(3*i,0.);
		std::vector<double> ub(3*i,0.);

		init_bounds(M, line,i,lb,ub);

		for(int j(0); j<i; j++) {
			for(int k(0); k<dim_v; k++) {			
				model_tab[k]+= model_function(k+1,params[3*j], params[1+3*j], params[2+3*j]);
			}
		}
		
		for(int p(0); p<dim_v; p++) {
			
			residual[p]=model_tab[p]-line[p];
		}
		
		std::vector<double> x(3*i,0.);

		for(int p(0); p<3*(i-1); p++){	
			x[p]=params[p];
		}
		/*
		for(int ind(0); ind<residual.size(); ind++){
			std::cout<<"residual["<<ind<<"] = "<<residual[ind]<<std::endl;
		}
		*/
		x[1+3*(i-1)] = minloc(residual);
		int rang = std::distance(residual.begin(), std::min_element( residual.begin()+1, residual.end() ));
		std::cout<<" argmin residual = "<< rang<<std::endl;
		std::cout<<" x["<<1+3*(i-1)<<"] = "<<x[1+3*(i-1)]<<std::endl;
		x[0+3*(i-1)] = line[int(x[1+3*(i-1)])]*M.amp_fact_init;
		x[2+3*(i-1)] = M.sig_init;
		
		for(int o(0); o<params.size();o++){
			std::cout<<"params["<<o<<"] = "<<params[o]<<std::endl;
		}

		std::cout<<"CALLING L-BFGS-B"<<std::endl;

		minimize_spec(M, 3*i, M.m, x, lb, ub, line);

		std::cout<<"DONE"<<std::endl;

		double val;

		for(int p(0); p<3*i; p++) {
			val = x[p];
//			std::cout << "1/////////ETAT fit_params.size() : " << M.fit_params.size() << " , " << M.fit_params[0].size() << " , " << M.fit_params[0][0].size() <<  std::endl;
			params[p]=val;
//			std::cout << "2/////////ETAT fit_params.size() : " << M.fit_params.size() << " , " << M.fit_params[0].size() << " , " << M.fit_params[0][0].size() <<  std::endl;
			std::cout<<"p = "<<p<<std::endl;
			std::cout<<"x["<<p << "] = "<<x[p]<<std::endl;
			std::cout<<"params["<<p << "] = "<<params[p]<<std::endl;
		}
		
	}
}


void algo_rohsa::mean_array(int power, std::vector<std::vector<std::vector<double>>> &mean_array_)
{	
	std::vector<double> spectrum(file.dim_cube[0],0.);
	int n = file.dim_cube[1]/power;
	std::cout << " n = " << n << std::endl;
	std::cout << " power = " << power << std::endl;
	//std::vector<std::vector<std::vector<double>>> cube_mean(file.dim_cube[0],std::vector<std::vector<double>>(power, std::vector<double>(power)));

	for(int i(0); i<power; i++)
	{
		for(int j(0); j<power; j++)
		{
			for(int k(0); k<n; k++)
			{
				for (int l(0); l<n; l++)
				{
					for(int m(0); m<file.dim_cube[0]; m++)
					{

//						std::cout<< "  test __  i,j,k,l,m,n ="<<i<<","<<j<<","<<k <<","<<l<<","<<m<<","<<n<< std::endl;
//						std::cout << "  test__ "<<k+j*n<<std::endl;
						spectrum[m] += file.cube[m][l+i*n][k+j*n];
					}
				}
			}
			for(int m(0); m<file.dim_cube[0]; m++)
			{
				/*cube_mean*/mean_array_[j][i][m] = spectrum[m]/pow(n,2);
			}
			for(int p(0); p<file.dim_cube[0]; p++)
			{
				spectrum[p] = 0.;
			}
		}
	}
/*
	for(int i(0); i<mean_array_[0][0].size(); i++){
		for(int j(0); j<mean_array_[0].size(); j++){
			for(int k(0); k<mean_array_.size(); k++){
				std::cout<<"cube_mean("<<i<<","<<j<<","<<k<<") = "<<mean_array_[i][j][k]<<std::endl;
			}
		}
	}
*/

}
/*    //PRINT CUBE_MEAN

	for(int i(0); i<power; i++)
	{
		for(int j(0); j<power; j++)
		{
			for(int k(0); k<file.dim_cube[2]; k++)
			{
//			std::cout<< "  test __  i,j,k,l,m,n ="<<i<<","<<j<<","<<k <<","<<l<<","<<m<<","<<n<< std::endl;
				std::cout << "  cube_mean("<<i<<","<<j<<","<<k<<") = "<<cube_mean[i][j][k]<<std::endl;
			}
		}
	}
*/


/*	//COMPARAISON
	for(int i(0); i<file.dim_cube[0]; i++)
	{
		for(int j(0); j<file.dim_cube[1]; j++)
		{
			for(int k(0); k<file.dim_cube[2]; k++)
			{
				//std::cout << "  cube("<<i<<","<<j<<","<<k<<") = "<<file.cube[i][j][k]<<std::endl;
				if(file.cube[i][j][k] - file.data[i][j][k] >1e-2)
				{ 
				std::cout << "  cube("<<i<<","<<j<<","<<k<<") = "<<file.cube[i][j][k]<<std::endl;
				std::cout << "  data("<<i<<","<<j<<","<<k<<") = "<<file.data[i][j][k]<<std::endl;
				file.cube[i][j][k] = file.data[i][j][k]:
				}
			}
		}
	}
*/




/*
    allocate(spectrum(size(cube,dim=1)))
    spectrum = 0.
    
    n = size(cube, dim=2) / nside

	print*," n =",n
	print*,"  size(cube, dim=1) = ",size(cube, dim=1)
	print*,"  size(cube, dim=2) = ",size(cube, dim=2)
	print*,"   nside = ", nside


    do i=1,size(cube_mean,dim=2)
       do j=1,size(cube_mean,dim=3)
          do k=1,n
             do l=1,n
                spectrum = spectrum + cube(:,k+((i-1)*n),l+((j-1)*n))
		print*, "__test__",k+(i-1)*n
             enddo
          enddo
          spectrum = spectrum / (n**2)
          cube_mean(:,i,j) = spectrum
          spectrum = 0.
       enddo
    enddo


*/

void algo_rohsa::convolution_2D_mirror(const std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &conv, int dim_y, int dim_x, int dim_k)
{
	int ii(0),jj(0),mm(0),nn(0),kCenterY(0), kCenterX(0);

	std::vector <std::vector<double>> ext_conv(dim_x+4, std::vector<double>(dim_y+4));
	std::vector <std::vector<double>> extended(dim_x+4, std::vector<double>(dim_y+4));

	for(int j(0); j<dim_x; j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			extended[2+i][2+j]=image[i][j];
		}
	}

	for(int j(0); j<2; j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			extended[2+i][j] = image[i][j];
		}
	}

	for(int i(0); i<2; i++)
	{
		for(int j(0); j<dim_x; j++)
		{
			extended[i][2+j] = image[i][j];
		}
	}

	for(int j(dim_x); j<dim_x+2; j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			extended[2+i][2+j]=image[i][j-2];
		}
	}

	for(int j(0); j<dim_x; j++)
	{
		for(int i(dim_y); i<dim_y+2; i++)
		{
			extended[2+i][2+j]=image[i-2][j];
		}
	}

	kCenterY = dim_k/2+1;
	kCenterX = kCenterY;

	/* //Afficher extended
	for(int i(0);i<dim_x+2;i++)
	{
		for(int j(0);j<dim_y+2;j++)
		{
			std::cout<<"extended["<<i<<"]["<<j<<"] = "<<extended[i][j]<<std::endl;
		}
	}
	std::cout<<"kCenterY = "<<kCenterY<<std::endl;
	*/


	for(int j(1);j<=dim_x+4;j++)
	{
		for(int i(1); i<=dim_y+4; i++)
		{
			for(int m(1); m<=dim_k ; m++)
			{
				mm = dim_k - m + 1;

				for(int n(1);n<=dim_k;n++)
				{
//					std::cout<<"ii = "<<ii<<"    jj = "<<jj<<std::endl;

					nn = dim_k - n + 1;

					ii = i + (m - kCenterY);
					jj = j + (n - kCenterX);

					if( ii >= 1 && ii < dim_y+4 && jj>=1 && jj< dim_x+4 )
					{
						std::cout<<"mm = "<<mm-1<<"    nn = "<<nn-1<<std::endl;
						std::cout<<"kernel["<<mm-1<<"]["<<nn-1<<"] = "<<kernel[mm-1][nn-1]<<std::endl;

						ext_conv[i-1][j-1] += extended[ii-1][jj-1]*kernel[mm-1][nn-1];
					}
				}
			}
		}
	}

	for(int j(0);j<dim_x;j++)
	{
		for(int i(0); i<dim_y; i++)
		{
			conv[i][j] = ext_conv[2+i][2+j];
		}
	}
/*
        for(int j(0); j<dim; j++)
	{
        	for (int k(0); k<dim[1]; k++)
		{
		        std::cout << " résultat["<<j<<"]["<<k<<"]= " << ext_conv[j][k] << std::endl;
        	}
        }
*/

}

// // L'ordre x,y,lambda est celui du code fortran : lambda,y,x      pk?

// It transforms a 1D vector into a contiguous flattened 1D array from a 2D array, like a valarray

void algo_rohsa::ravel_2D(const std::vector<std::vector<double>> &map, std::vector<double> &vector, int dim_y, int dim_x)
{
	int i__(0);

	for(int k(0); k<dim_x; k++)
	{
		for(int j(0);j<dim_y;j++)
		{
			vector[i__] = map[j][k];
			i__++;
		}
	}
}

// It transforms a 1D vector into a contiguous flattened 1D array from a 3D array, the interest is close to the valarray's one

void algo_rohsa::ravel_3D(const std::vector<std::vector<std::vector<double>>> &cube, std::vector<double> &vector, int dim_v, int dim_y, int dim_x)
{
        int i__(1);

        for(int k(0); k<dim_x; k++)
        {
                for(int j(0); j<dim_y; j++)
                {
			for(int i(0); i<dim_v; i++)
			{
	                        vector[i__] = cube[i][j][k];
        	                i__++;
			}
                }
        }
}


void algo_rohsa::ravel_3D_abs(const std::vector<std::vector<std::vector<double>>> &cube, const std::vector<std::vector<std::vector<double>>> &cube_abs, std::vector<double> &vector, int dim_v, int dim_y, int dim_x)
{
        int i__(1);

        for(int k(0); k<dim_x; k++)
        {
                for(int j(0); j<dim_y; j++)
                {
                        for(int i(0); i<dim_v; i++)
                        {
                                vector[i__] = cube[i][j][k];
                                i__++;
                        }
                }
        }

	for(int k(0); k<dim_x; k++)
        {
                for(int j(0); j<dim_y; j++)
                {
                        for(int i(0); i<dim_v; i++)
                   	{
                                vector[i__] = cube_abs[i][j][k];
                                i__++;
                        }
                }
        }
}


// It transforms a 1D vector into a 3D array, like the step we went through when analysing data from CCfits which returns a valarray that needs to be expended into a 3D array (it's the data cube)

void algo_rohsa::unravel_3D(const std::vector<double> &vector, std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x)
{
	int i__(1);

	for(int k(0); k<dim_x; k++)
	{
		for(int j(0); j<dim_y; j++)
		{
			for(int i(0); i<dim_v; i++)
			{
				cube[i][j][k] = vector[i__];
				i__++;
			}
		}
	}
}

void algo_rohsa::unravel_3D_abs(const std::vector<double> &vector, std::vector<std::vector<std::vector<double>>> &cube_abs,std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x)
{
        int i__(1);

        for(int k(0); k<dim_x; k++)
        {
                for(int j(0); j<dim_y; j++)
                {
                        for(int i(0); i<dim_v; i++)
                        {
                                cube[i][j][k] = vector[i__];
                                i__++;
                        }
                }
        }

        for(int k(0); k<dim_x; k++)
        {
                for(int j(0); j<dim_y; j++)
                {
                        for(int i(0); i<dim_v; i++)
                        {
                                cube_abs[i][j][k] = vector[i__];
                                i__++;
                        }
                }
        }
}

// It returns the mean value of a 1D vector
double algo_rohsa::mean(const std::vector<double> &array)
{
 	return std::accumulate(array.begin(), array.end(), 0.)/std::max(1.,double(array.size()));
}

// It returns the standard deviation value of a 1D vector
// BEWARE THE STD LIBRARY 
// "Std" rather than "std"

double algo_rohsa::Std(const std::vector<double> &array)
{
	double mean_(0.), var(0.);
	int n = array.size();
	mean_ = mean(array);

	for(int i(0); i<n; i++)
	{
		var+=pow(array[i]-mean_,2);
	}
	return sqrt(var/(n-1));
}

double algo_rohsa::std_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x)
{

	std::vector<double> vector(dim_x*dim_y, 0.);
	ravel_2D(map, vector, dim_y, dim_x);
	return Std(vector);
}


double algo_rohsa::max_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x)
{
	std::vector<double> vector(dim_x*dim_y,0.);
	ravel_2D(map, vector, dim_y, dim_x);
	double val_max = vector[0];
	for (unsigned int i = 0; i < vector.size(); i++)
		if (vector[i] > val_max)
    			val_max = vector[i];
	vector.clear();
	return val_max;
}

double algo_rohsa::mean_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x)
{
	std::vector<double> vector(dim_y*dim_x, 0.);
	ravel_2D(map, vector, dim_y, dim_x);
	double mean_2D = mean(vector);
	vector.clear();
	return mean_2D;
}

void algo_rohsa::std_spectrum(int dim_x, int dim_y, int dim_v)
{
	for(int i(0); i<dim_v; i++)
	{

		std::vector<std::vector<double>> map(dim_x, std::vector<double>(dim_y,0.));

		for(int j(0); j< dim_y; j++)
		{
			for( int k(0); k<dim_x; k++)
			{
				map[k][j] = file.data[k][j][i];
			}
		}		
		std_spect.vector::push_back(std_2D(map, dim_y, dim_x));
	}
}

void algo_rohsa::mean_spectrum(int dim_x, int dim_y, int dim_v)
{

	for(int i(0);i<dim_v;i++)
	{
		std::vector<std::vector<double>> map(dim_x, std::vector<double>(dim_y, 0.));
		for(int j(0); j<dim_y ; j++)
		{
			for(int k(0); k<dim_x ; k++)
			{
				map[k][j] = file.data[k][j][i];
			}
		}

		mean_spect.vector::push_back(mean_2D(map, dim_y, dim_x));
		map.clear();
	}
}

void algo_rohsa::max_spectrum(int dim_x, int dim_y, int dim_v)
{
	for(int i(0); i<dim_v; i++)
	{
		std::vector<std::vector<double>> map(dim_x, std::vector<double>(dim_y,0.));
		for(int j(0); j< dim_y; j++)
		{
			for( int k(0); k<dim_x; k++)
			{
				map[k][j] = file.data[k][j][i];
			}
		}
		max_spect.vector::push_back(max_2D(map, dim_y, dim_x));
		map.clear();
	}
}

void algo_rohsa::max_spectrum_norm(int dim_x, int dim_y, int dim_v, double norm_value)
{
	for(int i(0); i<dim_v; i++)
	{
		std::vector<std::vector<double>> map(dim_x, std::vector<double>(dim_y));
		for(int j(0); j< dim_y; j++)
		{
			for( int k(0); k<dim_x; k++)
			{
				map[k][j] = file.data[k][j][i];
			}
		}
		max_spect_norm.vector::push_back(max_2D(map, dim_y, dim_x));
		map.clear();
	}

	double val_max = max_spect_norm[0];
	for (unsigned int i = 0; i < max_spect_norm.size(); i++)
		if (max_spect_norm[i] > val_max)
    			val_max = max_spect_norm[i];

	for(int i(0); i<dim_v ; i++)
	{
		max_spect_norm[i] /= val_max/norm_value; 
	}
}

