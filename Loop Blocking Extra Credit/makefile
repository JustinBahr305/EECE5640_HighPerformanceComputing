all: MM_openMP MM_block MM_no_block MM_no_block_openMP

MM_openMP:
	g++ -fopenmp -o MM_openMP MM_openMP.cpp

MM_block:
	g++ -o MM_block MM_block.cpp

MM_no_block:
	g++ -o MM_no_block MM_no_block.cpp

MM_no_block_openMP:
	g++ -fopenmp -o MM_no_block_openMP MM_no_block_openMP.cpp

clean:
	rm MM_openMP MM_block MM_no_block MM_no_block_openMP