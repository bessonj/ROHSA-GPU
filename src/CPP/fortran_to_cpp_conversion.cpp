#include "fortran_to_cpp_conversion.h"

void ConvertToFortran(char* fstring, std::size_t fstring_len,
                      const char* cstring)
{
    std::size_t inlen = std::strlen(cstring);
    std::size_t cpylen = std::min(inlen, fstring_len);
    if (inlen > fstring_len)
    {
		std::cout<<"LENGTH ERROR IN FORTRAN CONVERSION OF CHAR"<<std::endl; 
        // TODO: truncation error or warning
    }
    std::copy(cstring, cstring + cpylen, fstring);
    std::fill(fstring + cpylen, fstring + fstring_len, ' ');
}


//extern void test();
int minimize() {
    /* System generated locals */
    int i__1;
    double d__1, d__2;
    /* Local variables */
    double f, g[25];
    static int i__;
    static double l[25];
    static int m, n;
    double u[25], x[25], t1, t2, wa[690];
    static int nbd[25], iwa[75];

    static char task_c[60] = {'S','T','A','R','T', '\0'};
    static char task_FG_c[60] = "FG";  
    static char task_FG_START_c[60] = "FG_START";  
    static char task_NEW_X_c[60] = "NEW_X";  
    static char task_STOP_c[60] = "STOP";  
    static char task_STOP_GRAD_c[60] = "STOP_GRAD";  
    static char task_START_c[60] = "START";  

	char task[60];
    ConvertToFortran(task, sizeof task_c, task_c);
	std::cout<<"===> task_c = "<<task_c <<std::endl;

/*    static char taskValue;
    static char *task[]=&taskValue; 
*/
/*      http://stackoverflow.com/a/11278093/269192 */
    static double factr;
/*     static char csave[60]; */
/*
    static char csaveValue;
    char *csave=&csaveValue;
*/
    static char csaveValue;
    static char csave[60];

    static double dsave[29];
    int isave[44];
    int lsave[4];
    double pgtol;
    static int iprint;


/*
    This simple driver demonstrates how to call the L-BFGS-B code to 
      solve a sample problem (the extended Rosenbrock function 
      subject to bounds on the variables). The dimension n of this 
      problem is variable. 
       nmax is the dimension of the largest problem to be solved. 
       mmax is the maximum number of limited memory corrections. 
    Declare the variables needed by the code. 
      A description of all these variables is given at the end of 
      the driver. 
    Declare a few additional variables for this sample problem. 
 */

/*     We wish to have output at every iteration. */
    iprint = 1; 
/*     iprint = 101; */
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
    for (i__ = 1; i__ <= i__1; i__ += 2) {
        nbd[i__ - 1] = 2;
        l[i__ - 1] = 1.;
        u[i__ - 1] = 100.;
    }
    /*     Next set bounds on the even-numbered variables. */
    i__1 = n;
    for (i__ = 2; i__ <= i__1; i__ += 2) {
        nbd[i__ - 1] = 2;
        l[i__ - 1] = -100.;
        u[i__ - 1] = 100.;
    }
    /*     We now define the starting point. */
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
        x[i__ - 1] = 3.;
        /* L14: */
    }
    printf("     Solving sample problem (Rosenbrock test fcn).\n");
    printf("      (f = 0.0 at the optimal solution.)\n");

    /*     We start the iteration by initializing task. */

//    task = 'START';
/*     s_copy(task, "START", (ftnlen)60, (ftnlen)5); */
    /*        ------- the beginning of the loop ---------- */
L111:
    /*     This is the call to the L-BFGS-B code. */
//    setulb(&n, &m, x, l, u, nbd, &f, g, &factr, &pgtol, wa, iwa, task, &iprint, csave, lsave, isave, dsave);
//    setulb_(&n, &m, x, l, u, nbd, &f, g, &factr, &pgtol, wa, iwa, task, &iprint, csave, lsave, isave, dsave);

	for(int p=0; p<n; p++){
		printf("->x[%d] = %f\n", p, x[p]);
	}

/////////
    char cstring[2][4] = { "abc", "xyz" };
    char string[2][4];

    ConvertToFortran(string[0], sizeof string[0], cstring[0]);
    ConvertToFortran(string[1], sizeof string[1], cstring[1]);

    std::cout << "c++: string[0] = '" << cstring[0] << "'" << std::endl;
    std::cout << "c++: string[1] = '" << cstring[1] << "'" << std::endl;
/////////
//	exit(0);
/////////

	std::cout<<"===> task = "<<task <<std::endl;
//	std::cout<<"===> len(task) = "<<task <<std::endl;
//	std::cout<<"===> task==START = "<< task=="START" <<std::endl;
	std::cout<<"===> std::strncmp(task, START, 5) == 0 : "<< (std::strncmp(task, "START", 5) == 0) <<std::endl;

    setulb_(&n, &m, x, l, u, nbd, &f, g, &factr, &pgtol, wa, iwa, task, &iprint, csave, lsave, isave, dsave);
//	setulb_(n,m, double* x, double* l, double* u, int* nbd, double* f, double* g, double* factr, double* pgtol, double* wa, int* iwa, char *task, int* iprint, char *csave, bool *lsave, int *isave, double *dsave);
	std::cout<<"===> task = "<<task <<std::endl;

	for(int p=0; p<n; p++){
		printf("-->x[%d] = %f\n", p, x[p]);
	}

/*
	std::cout<<"compare task and FG_START = "<< std::strncmp(task, task_FG_START, 8)<<std::endl;
	std::cout<<"compare task and FG = "<< std::strncmp(task, task_FG, 2)<<std::endl;
	std::cout<<"compare a and a = "<< std::strncmp("aaaxcpvjà", "aaaspd,fp", 3)<<std::endl;
	exit(0);
*/
/*     if (s_cmp(task, "FG", (ftnlen)2, (ftnlen)2) == 0) { */
    if (std::strncmp(task, "FG", 2) == 0 || std::strncmp(task, "FG_START", 8) == 0 ){//task == 'FG' || task == 'FG_START'){ //IS_FG(*task) ) {
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
		for(int p=0; p<n; p++){
			printf("g[%d] = %f\n", p, g[p]);
		}
		printf("f = %f\n", f);

        /*          go back to the minimization routine. */
        goto L111;
    }

/*     if (s_cmp(task, "NEW_X", (ftnlen)5, (ftnlen)5) == 0) { */
    if ( std::strncmp(task, "NEW_X", 5) == 0){//task=='NEW_X' ) {
        goto L111;
    }
    /*        the minimization routine has returned with a new iterate, */
    /*         and we have opted to continue the iteration. */
    /*           ---------- the end of the loop ------------- */
    /*     If task is neither FG nor NEW_X we terminate execution. */
    //s_stop("", (ftnlen)0);

	for(int p=0; p<n; p++){
		printf("x[%d] = %.20f\n", p, x[p]);
	}

    return 0;
} /* MAIN__ */
