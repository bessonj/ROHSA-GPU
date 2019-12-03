CC=g++
CFLAGS= -fopenmp -L/usr/lib/x86_64-linux-gnu/ -std=c++11  
LDFLAGS= 
SOURCES= main.cpp hypercube.cpp model.cpp algo_rohsa.cpp ./L-BFGS-B-C/src/*.c
EXECNAME= EXEC

all:
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(EXECNAME) $(SOURCES) -lCCfits -lcfitsio

clean:
	rm -f *.o core $(EXECNAME)



