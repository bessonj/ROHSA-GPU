#ifndef DEF_GAUSSIAN
#define DEF_GAUSSIAN

#include "Gaussian.h"
#include <omp.h>


Gaussian::Gaussian(const Parse &file1)
{
	n_gauss_add = 0;
	file = file1;

	std::string txt, egal;
        std::ifstream fichier("parameters.txt", std::ios::in);  // on ouvre en lecture
 
        if(fichier)  // si l'ouverture a fonctionné
        {
		fichier >> txt >> egal >> filename;
		fichier >> txt >> egal >> fileout;
		fichier >> txt >> egal >> filename_noise;
		fichier >> txt >> egal >> n_gauss;
		fichier >> txt >> egal >> lambda_amp;
		fichier >> txt >> egal >> lambda_mu;
		fichier >> txt >> egal >> lambda_sig;
		fichier >> txt >> egal >> lambda_var_amp;
		fichier >> txt >> egal >> lambda_var_mu;
		fichier >> txt >> egal >> lambda_var_sig;
		fichier >> txt >> egal >> amp_fact_init;
		fichier >> txt >> egal >> sig_init;
		fichier >> txt >> egal >> init_option;
		fichier >> txt >> egal >> maxiter_init;
		fichier >> txt >> egal >> maxiter;
		fichier >> txt >> egal >> m;
		fichier >> txt >> egal >> check_noise;
		fichier >> txt >> egal >> check_regul;
		fichier >> txt >> egal >> check_descent;
		fichier >> txt >> egal >> lstd;
		fichier >> txt >> egal >> ustd;
		fichier >> txt >> egal >> iprint;
		fichier >> txt >> egal >> iprint_init;
		fichier >> txt >> egal >> check_save_grid;

		if(check_save_grid == "true")
			save_grid = true;
		else 
			save_grid = false;		
		if(check_noise == "true")
			noise = true;
		else
			noise = false;
		if(check_regul == "true")
			regul = true;
		else
			regul = false;
		if(check_descent == "true")
			descent = true;
		else
			descent = false;

                fichier.close();
        }
        else
                std::cerr << "Impossible d'ouvrir le fichier !" << std::endl;

	std::vector<double> slice(3,0.);
	kernel.vector::push_back(slice);
	kernel.vector::push_back(slice);
	kernel.vector::push_back(slice);
	slice.clear();

	kernel[0][0] = 0.;
	kernel[0][1] = -0.25;
	kernel[0][2] = 0.;
	kernel[1][0] = -0.25;
	kernel[1][1] = 1.;
	kernel[1][2] = -0.25;
	kernel[2][0] = 0.;
	kernel[2][1] = -0.25;
	kernel[2][2] = 0.;

	dim_data = file.get_dim_data();
	dim_x = dim_data[2];
	dim_y = dim_data[1];
	dim_v = dim_data[0];

	std_spectrum(dim_x, dim_y, dim_v); //oublier
	mean_spectrum(dim_x, dim_y, dim_v);
	max_spectrum(dim_x, dim_y, dim_v); //oublier

	double max_mean_spect = mean_spect[0];
	for (unsigned int i = 0; i < mean_spect.size(); i++)
		if (mean_spect[i] > max_mean_spect)
    			max_mean_spect = mean_spect[i];

	max_spectrum_norm(dim_x,dim_y, dim_v, max_mean_spect);
/*
	for(int l(0); l<dim_v; l++)
	{
		std::cout<<"max_spect_norm["<<l<<"] = "<<max_spect_norm[l]<<std::endl;
	}
*/
	std::cout << " descent : "<< descent << std::endl;

	if(descent)
	{		
//		std::vector<std::vector<std::vector<double>>> grid_params(3*(n_gauss+(file.nside*n_gauss_add)), std::vector<std::vector<double>>(dim_y, std::vector<double>(dim_x)));
//		std::vector<std::vector<std::vector<double>>> fit_params(3*(n_gauss+(file.nside*n_gauss_add)), std::vector<std::vector<double>>(1, std::vector<double>(1,1.)));

		std::vector<std::vector<double>> sheet(dim_y,std::vector<double>(dim_x));
		std::vector<std::vector<double>> sheet_(1,std::vector<double>(1,0.));

		std::cout<<"nside = "<< file.nside<<std::endl;

		for(int compteur(0) ; compteur< 3*(n_gauss+(file.nside*n_gauss_add)); compteur++)
		{
			grid_params.vector::push_back(sheet);
		}
		for(int compteur(0); compteur<3*(n_gauss+(file.nside*n_gauss_add)); compteur++) {
			fit_params.vector::push_back(sheet_);
		}
		sheet.clear();
		sheet_.clear();

//		std::cout << "test fit_params : "<<fit_params[0][0][0]<<std::endl;


		for(int i(0);i<n_gauss; i++){
			std::cout<<"TEST"<<std::endl;
			fit_params[0+3*i][0][0] = 0.;
			fit_params[1+3*i][0][0] = 1.;
			fit_params[2+3*i][0][0] = 1.;
		}
		for(int i(0); i<fit_params.size();i++){
			std::cout<<"fit_params["<<i<<"][0][0] = "<<fit_params[i][0][0]<<std::endl;
		}

	} else {
//		std::vector<std::vector<std::vector<double>>> fit_params(3*(n_gauss+n_gauss_add), std::vector<std::vector<double>>(dim_y, std::vector<double>(dim_x)));
				
		std::vector<std::vector<double>> sheet(dim_y,std::vector<double>(dim_x));
		for(int compteur(0); compteur<3*(n_gauss+n_gauss_add) ; compteur++)
		{
			grid_params.vector::push_back(sheet);
		}

		sheet.clear();

	}


	std::cout << "fit_params.size() : " << fit_params.size() << " , " << fit_params[0].size() << " , " << fit_params[0][0].size() <<  std::endl;
	std::cout << "grid_params.size() : "<< grid_params.size() << " , " << grid_params[0].size()  << " , " << grid_params[0][0].size() << std::endl;

	if(descent)
	{
		for(int n(0); n<file.nside; n++)
		{
			int power(pow(2,n));

			
			std::cout << " power = " << power << std::endl;
			std::cout << " cube "<<file.cube[0][1][0]<<std::endl;

			std::vector<std::vector<std::vector<double>>> cube_mean(dim_v,std::vector<std::vector<double>>(power,std::vector<double>(power,1.)));

			mean_array(power, cube_mean);

			std::vector<double> cube_mean_flat(cube_mean.size());
			std::vector<double> params(fit_params[0][0].size(),0.);

			for(int e(0); e<fit_params[0][0].size(); e++) {
				params[e] = fit_params[0][0][e];
			}
			for(int e(0); e<cube_mean.size(); e++) {
				cube_mean_flat[e] = cube_mean[0][0][e];
			}
			if (n==0)
			{
				//assume option "mean"
				std::cout<<"Init mean spectrum"<<std::endl;
				
				init_spectrum(sig_init, cube_mean_flat, params);	
			}
		}
	}

/*
    if (descent .eqv. .true.) then
       print*, "Start hierarchical descent"
       !Start iteration
       do n=0,nside-1
          power = 2**n
          allocate(cube_mean(dim_cube(1), power, power))

          call mean_array(power, cube, cube_mean)          
          if (n == 0) then
             if (init_option .eq. "mean") then
                print*, "Init mean spectrum"        
                call init_spectrum(n_gauss, fit_params(:,1,1), dim_cube(1), cube_mean(:,1,1), amp_fact_init, sig_init, &
                     ub_sig_init, maxiter_init, m, iprint_init)
             elseif (init_option .eq. "std") then
                call init_spectrum(n_gauss, fit_params(:,1,1), dim_cube(1), std_spect, amp_fact_init, sig_init, &
                     ub_sig_init, maxiter_init, m, iprint_init)
             elseif (init_option .eq. "max") then
                call init_spectrum(n_gauss, fit_params(:,1,1), dim_cube(1), max_spect, amp_fact_init, sig_init, &
                     ub_sig_init, maxiter_init, m, iprint_init)
             elseif (init_option .eq. "maxnorm") then
                call init_spectrum(n_gauss, fit_params(:,1,1), dim_cube(1), max_spect_norm, amp_fact_init, sig_init, &
                     ub_sig_init, maxiter_init, m, iprint_init)
             else 
                print*, "init_option keyword should be 'mean' or 'std' or 'max' or 'maxnorm'"
                stop
             end if
          end if
*/ 
}

void Gaussian::init_bounds(std::vector<double> line, int n_gauss_local, std::vector<double> lb, std::vector<double> ub, double ub_sig) {

	double max_line = *std::max_element(line.begin(), line.end());

	for(int i(0); i<n_gauss_local; i++) {

		lb[0+3*i]=0.;
		ub[0+3*i]=max_line;

		lb[1+3*i]=0.;
		ub[1+3*i]=dim_v;


		lb[2+3*i]=0.0001;
		ub[2+3*i]=ub_sig;
			

	}
}

double Gaussian::gaussian(int x, double a, double m, double s) {

	return a*exp(-pow((double(x)-m),2.) / (2.*pow(s,2.)));

}

int Gaussian::minloc(std::vector<double> tab) {
	return std::distance(tab.begin(), std::min_element( tab.begin()+1, tab.end() ));
}


int Gaussian::minimize_spec(long n, long m, std::vector<double> x_v, std::vector<double> lb_v, std::vector<double> ub_v, std::vector<double> line_v) {
/* Minimize_spec */ 
//int MAIN__(void)
    /* System generated locals */
    integer i__1;
    double d__1, d__2;
    /* Local variables */
    static double f, g[1024];
    static integer i__;
    static double t1, t2, wa[43251];
    static integer nbd[1024], iwa[3072];
/*     static char task[60]; */
    static integer taskValue;
    static integer *task=&taskValue; /* must initialize !! */
/*      http://stackoverflow.com/a/11278093/269192 */
    static double factr;
/*     static char csave[60]; */
    static integer csaveValue;
    static integer *csave=&csaveValue;
    static double dsave[29];
    static integer isave[44];
    static logical lsave[4];
    static double pgtol;
    static integer iprint;

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
/*     We specify the dimension n of the sample problem and the number */
/*        m of limited memory corrections stored.  (n and m should not */
/*        exceed the limits nmax and mmax respectively.) */
    n = 25;
    m = 5;
/*     We now provide nbd which defines the bounds on the variables: */
/*                    l   specifies the lower bounds, */
/*                    u   specifies the upper bounds. */
/*     First set bounds on the odd-numbered variables. */
    i__1 = n;
    for (i__ = 0; i__ < i__1; i__ ++) {
        nbd[i__] = 2;
    }

    printf("     Solving sample problem (Rosenbrock test fcn).\n");
    printf("      (f = 0.0 at the optimal solution.)\n");

    /*     We start the iteration by initializing task. */

    *task = (integer)START;
/*     s_copy(task, "START", (ftnlen)60, (ftnlen)5); */
    /*        ------- the beginning of the loop ---------- */
L111:
    /*     This is the call to the L-BFGS-B code. */
    setulb(&n, &m, x, lb, ub, nbd, &f, g, &factr, &pgtol, wa, iwa, task, 
            &iprint, csave, lsave, isave, dsave);
/*     if (s_cmp(task, "FG", (ftnlen)2, (ftnlen)2) == 0) { */
    if ( IS_FG(*task) ) {
        /*        the minimization routine has returned to request the */
        /*        function f and gradient g values at the current x. */
        /*        Compute function value f for the sample problem. */
        /* Computing 2nd power */
        d__1 = x[0] - 1.;
        f = d__1 * d__1 * .25;
        i__1 = n;
        for (i__ = 2; i__ <= i__1; ++i__) {
            /* Computing 2nd power */
            d__2 = x[i__ - 2];
            /* Computing 2nd power */
            d__1 = x[i__ - 1] - d__2 * d__2;
            f += d__1 * d__1;
        }
        f *= 4.;
        /*        Compute gradient g for the sample problem. */
        /* Computing 2nd power */
        d__1 = x[0];
        t1 = x[1] - d__1 * d__1;
        g[0] = (x[0] - 1.) * 2. - x[0] * 16. * t1;
        i__1 = n - 1;
        for (i__ = 2; i__ <= i__1; ++i__) {
            t2 = t1;
            /* Computing 2nd power */
            d__1 = x[i__ - 1];
            t1 = x[i__] - d__1 * d__1;
            g[i__ - 1] = t2 * 8. - x[i__ - 1] * 16. * t1;
            /* L22: */
        }
        g[n - 1] = t1 * 8.;
        /*          go back to the minimization routine. */
        goto L111;
    }

/*     if (s_cmp(task, "NEW_X", (ftnlen)5, (ftnlen)5) == 0) { */
    if ( *task==NEW_X ) {
        goto L111;
    }
    /*        the minimization routine has returned with a new iterate, */
    /*         and we have opted to continue the iteration. */
    /*           ---------- the end of the loop ------------- */
    /*     If task is neither FG nor NEW_X we terminate execution. */
    //s_stop("", (ftnlen)0);
    return 0;

}

void Gaussian::init_spectrum(double ub_sig, std::vector<double> line, std::vector<double> params) {

	std::vector<double> model(dim_v,0.);
	std::vector<double> residual(dim_v,0.);

	for(int i(1); i<=n_gauss; i++) {

		std::vector<double> lb(3*i,0.);
		std::vector<double> ub(3*i,0.);
		
		init_bounds(line,i,lb,ub,ub_sig);

		for(int j(0); j<i; j++) {
			for(int k(0); k<dim_v; k++) {			
				model[k]+= gaussian(k,params[3*j], params[1+3*j], params[2+3*j]);
			}
		}
		
		for(int p(0); p<dim_v; p++) {
			
			residual[p]=model[p]-line[p];
		}
		
		std::vector<double> x(3*i,0.);

		for(int p(0); p<3*(i-1); p++){	
			x[p]=params[p];
		}

		for(int ind(0); ind<residual.size(); ind++){
			std::cout<<"residual["<<ind<<"] = "<<residual[ind]<<std::endl;
		}

		x[1+3*(i-1)] = minloc(residual);
		int rang = std::distance(residual.begin(), std::min_element( residual.begin()+1, residual.end() ));
		std::cout<<" argmin residual = "<< rang<<std::endl;
		std::cout<<" TEST "<<int(x[1+3*(i-1)])<<","<<x[1+3*(i-1)]<<std::endl;
		x[0+3*(i-1)] = line[int(x[1+3*(i-1)])]*amp_fact_init;
		std::cout<<" TEST "<<std::endl;
		x[2+3*(i-1)] = sig_init;

		for(int o(0); o<params.size();o++){
			std::cout<<"params["<<o<<"] = "<<params[o]<<std::endl;
		}

		minimize_spec(3*i, m, x, lb, ub, line);

		for(int p(0); p<3*i; p++) {
			params[p]= x[p];
		}
	}
}


void Gaussian::mean_array(int power, std::vector<std::vector<std::vector<double>>> &mean_array_)
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
				/*cube_mean*/mean_array_[m][j][i] = spectrum[m]/pow(n,2);
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

void Gaussian::convolution_2D_mirror(const std::vector<std::vector<double>> &image, std::vector<std::vector<double>> &conv, int dim_y, int dim_x, int dim_k)
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

void Gaussian::ravel_2D(const std::vector<std::vector<double>> &map, std::vector<double> &vector, int dim_y, int dim_x)
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

void Gaussian::ravel_3D(const std::vector<std::vector<std::vector<double>>> &cube, std::vector<double> &vector, int dim_v, int dim_y, int dim_x)
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


void Gaussian::ravel_3D_abs(const std::vector<std::vector<std::vector<double>>> &cube, const std::vector<std::vector<std::vector<double>>> &cube_abs, std::vector<double> &vector, int dim_v, int dim_y, int dim_x)
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

void Gaussian::unravel_3D(const std::vector<double> &vector, std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x)
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

void Gaussian::unravel_3D_abs(const std::vector<double> &vector, std::vector<std::vector<std::vector<double>>> &cube_abs,std::vector<std::vector<std::vector<double>>> &cube, int dim_v, int dim_y, int dim_x)
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
double Gaussian::mean(const std::vector<double> &array)
{
 	return std::accumulate(array.begin(), array.end(), 0.)/std::max(1.,double(array.size()));
}

// It returns the standard deviation value of a 1D vector
// BEWARE THE STD LIBRARY 
// "Std" rather than "std"

double Gaussian::Std(const std::vector<double> &array)
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

double Gaussian::std_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x)
{

	std::vector<double> vector(dim_x*dim_y, 0.);
	ravel_2D(map, vector, dim_y, dim_x);
	return Std(vector);
}


double Gaussian::max_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x)
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

double Gaussian::mean_2D(const std::vector<std::vector<double>> &map, int dim_y, int dim_x)
{
	std::vector<double> vector(dim_y*dim_x, 0.);
	ravel_2D(map, vector, dim_y, dim_x);
	double mean_2D = mean(vector);
	vector.clear();
	return mean_2D;
}

void Gaussian::std_spectrum(int dim_x, int dim_y, int dim_v)
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

void Gaussian::mean_spectrum(int dim_x, int dim_y, int dim_v)
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

void Gaussian::max_spectrum(int dim_x, int dim_y, int dim_v)
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

void Gaussian::max_spectrum_norm(int dim_x, int dim_y, int dim_v, double norm_value)
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

/*    real(xp), intent(inout), dimension(:), allocatable :: spectrum !! max_spectrum of the observation
    real(xp), dimension(:,:), allocatable :: map !! 2D array

    integer :: i !! loop index

    do i=1,dim_v
       allocate(map(dim_y,dim_x))
       map = data(i,:,:)
       spectrum(i) = max_2D(map, dim_y, dim_x) 
       deallocate(map)
    end do

    if (present(norm_value)) then
       spectrum = spectrum / (maxval(spectrum) / norm_value)
    end if   
  end subroutine max_spectrum  
*/






#endif
