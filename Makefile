libocl_simple.so: ocl_simple.o
	g++ -g -shared ocl_simple.o -o libocl_simple.so

ocl_simple.o: ocl_simple.cpp ocl_simple.h
	g++ -g -c ocl_simple.cpp -o ocl_simple.o

test: test.cpp ocl_simple.h
	g++ -g test.cpp -lOpenCL -o test