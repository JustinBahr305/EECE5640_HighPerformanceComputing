all: Q1_omp Q1_leibniz Q1_pthread

Q1_omp:
	g++ -fopenmp -o Q1_omp Q1_omp.cpp

Q1_leibniz:
	g++ -fopenmp -o Q1_leibniz Q1_leibniz.cpp

Q1_pthread:
	g++ -lpthread -o Q1_pthread Q1_pthread.cpp

clean:
	rm Q1_omp Q1_leibniz Q1_pthread