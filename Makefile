all: libocl_simple.so libglobal_ocl_env.so tests test test_global 

libocl_simple.so: ocl_simple.o
	g++ -g -shared ocl_simple.o -o libocl_simple.so

ocl_simple.o: ocl_simple.cpp ocl_simple.h
	g++ -g -c -fPIC ocl_simple.cpp -o ocl_simple.o

libglobal_ocl_env.so: global_ocl_env.h global_ocl_env.cpp ocl_simple.h
	g++ -g -shared -fPIC global_ocl_env.cpp -L/opt/rocm/lib -lOpenCL -o libglobal_ocl_env.so

.PHONY: tests
tests:
	cd tests && make

clean:
	rm -f test test_global *.o *.so
	cd tests && make clean
