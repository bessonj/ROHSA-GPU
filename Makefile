CC=g++
CFLAGS= -O3 -fopenmp -L/usr/lib/x86_64-linux-gnu/ -std=c++11  
LDFLAGS= 
SOURCES= main.cpp Gaussian.cpp Parse.cpp /home/bematiste/CPP/ROHSA-GPU_test/L-BFGS-B-C/src/*.c
EXECNAME= EXEC

all:
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(EXECNAME) $(SOURCES) -lCCfits -lcfitsio

clean:
	rm -f *.o core $(EXECNAME)



