CC=g++
CFLAGS= -O3 -fopenmp -L/usr/lib/x86_64-linux-gnu/ -I/usr/include/python2.7/ -std=c++11  
LDFLAGS= 
SOURCES= main.cpp hypercube.cpp model.cpp algo_rohsa.cpp ./L-BFGS-B-C/src/*.c  
EXECNAME= EXEC

all:
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(EXECNAME) $(SOURCES) -lCCfits -lcfitsio -lpython2.7 

clean:
	rm -f *.o core $(EXECNAME)



