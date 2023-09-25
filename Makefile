libocl_simple.so: ocl_simple.o
	g++ -g -shared ocl_simple.o -o libocl_simple.so

ocl_simple.o: ocl_simple.cpp ocl_simple.h
	g++ -g -c ocl_simple.cpp -o ocl_simple.o

test: test.cpp ocl_simple.h
	g++ -g test.cpp -lOpenCL -o test

libglobal_ocl_env.so: global_ocl_env.h global_ocl_env.cpp ocl_simple.h
	g++ -g -shared -fPIC global_ocl_env.cpp -o libglobal_ocl_env.so

test_global: test_global.cpp libglobal_ocl_env.so
	g++ -g test_global.cpp -L. -lglobal_ocl_env -lOpenCL -o test_global

clean:
	rm -f test test_global *.o *.so
