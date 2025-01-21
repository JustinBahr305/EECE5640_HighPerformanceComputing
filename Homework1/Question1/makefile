all: Q1_floatbench Q1_floatO2 Q1_matrixbench Q1_matrixO2 Q1_intbench Q1_intO2

Q1_floatbench: Q1_floatbench.cpp
	g++ -Wall -g -o Q1_floatbench Q1_floatbench.cpp

Q1_floatO2: Q1_floatbench.cpp
	g++ -Wall -g -O2 -o Q1_floatO2 Q1_floatbench.cpp

Q1_matrixbench: Q1_matrixbench.cpp
	g++ -Wall -g -o Q1_matrixbench Q1_matrixbench.cpp

Q1_matrixO2: Q1_matrixbench.cpp
	g++ -Wall -g -O2 -o Q1_matrixO2 Q1_matrixbench.cpp

Q1_intbench: Q1_intbench.cpp
	g++ -Wall -g -o Q1_intbench Q1_intbench.cpp

Q1_intO2: Q1_intbench.cpp
	g++ -Wall -g -O2 -o Q1_intO2 Q1_intbench.cpp

clean:
	rm Q1_floatbench Q1_matrixbench Q1_intbench