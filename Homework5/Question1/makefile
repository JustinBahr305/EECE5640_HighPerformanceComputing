TARGET ?= Q1_gpu

all: $(TARGET)

Q1_gpu:
	nvcc -o Q1_gpu Q1_gpu.cpp hist.cu

Q1_omp:
	g++ -fopenmp -o Q1_omp Q1_omp.cpp

clean:
	rm -f Q1_gpu Q1_omp