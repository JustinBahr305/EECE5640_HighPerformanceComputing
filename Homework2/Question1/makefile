all: Q1_omp Q1_leibniz_omp Q1_pthread Q1_leibniz_pthread

Q1_omp:
	g++ -fopenmp -o Q1_omp Q1_omp.cpp

Q1_leibniz_omp:
	g++ -fopenmp -o Q1_leibniz_omp Q1_leibniz_omp.cpp

Q1_pthread:
	g++ -lpthread -o Q1_pthread Q1_pthread.cpp

Q1_leibniz_pthread:
	g++ -lpthread -o Q1_leibniz_pthread Q1_leibniz_pthread.cpp

clean:
	rm Q1_omp Q1_leibniz_omp Q1_pthread Q1_leibniz_pthread