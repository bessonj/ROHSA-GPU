I. Instructions before use :

	In order to satisfy the requirements below, 3 unix scripts are given :
	    install_L-BFGS-B.sh
	    install_CCfits.sh
	    install_matplotlib.sh
	(in some cases chmod +x *.sh could be required)

	Requirements :

	  -  cfitsio
	  -  CCfits
	  -  L-BFGS-B-C (Stephen Becker)
	  -  matplotlibcpp (Benno Evers)

	  And the following packages :

	    -  python-matplotlib 
	    -  python-numpy 
	    -  python2.7-dev

	You may also manually check whether or not cmake and libomp-dev (openMP) are installed.

II. Compiling and launching the program :

	A cmake file is written in the root directory, you can produce the Makefile in this directory with :

		cmake .

	Then,

		make 

	You will get a binary file, ROHSA-GPU

		./ROHSA-GPU

II. Instructions for use (WARNING) :

	parameters.txt needs to be modified with respect to the hypercube file

	You may put a at least a .fits or .dat file and specify true or false for the criteria below

		filename_dat = ./GHIGLS_DFN_NH.dat  <-- the location of the .dat file with respect to the root ./
		filename_fits = ./GHIGLS_DFN_Tb.fits  <-- the location of the .fits file with respect to the root .
		file_type_is_dat = true <-- the file is a .dat one
		file_type_is_fits = false <-- the file is a .fits one
		fileout = GHIGLS_DFN_Tb_gauss_run_0 <-- the outpout binary result
		filename_noise =  '' <-- the name of the noise file
		n_gauss =  3 <-- number of gaussians to be considered
		lambda_amp =  1. 
		lambda_mu =  1.
		lambda_sig =  1.
		lambda_var_amp =  0.
		lambda_var_mu =  0.
		lambda_var_sig =  1.0
		amp_fact_init =  0.66
		sig_init =  5.0
		init_option =  mean
		maxiter_init =  15000
		maxiter =  800 <-- maximum of iterations for BFGS
		m =  10 <--  limited memory corrections stored (L from L-BFGS)
		noise =  false  <-- noise activated
		regul =  true  <-- regulation activated
		descent =  true <-- descent activated
		lstd =  0
		ustd =  19
		iprint =  -1
		iprint_init =  -1
		save_grid =  false
		ub_sig = 100.0
		lb_sig = 0.001
		ub_sig_init = 1.
		lb_sig_init = 100.
