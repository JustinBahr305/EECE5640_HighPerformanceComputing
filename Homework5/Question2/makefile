all: Q2 Q2_tiled

Q2:
	nvcc -o Q2 Q2.cpp stencil.cu

Q2_tiled:
	nvcc -o Q2_tiled Q2.cpp tiled_stencil.cu

clean:
	rm Q2 Q2_tiled