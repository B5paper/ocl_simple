default: test test_global test_simple_ocl

rebuild:
	cd .. && make
	make

test: test.cpp
	g++ -g test.cpp -I.. -L/opt/rocm/lib -lOpenCL -o test

test_global: test_global.cpp
	g++ -g test_global.cpp -I.. -L.. -lglobal_ocl_env -L/opt/rocm/lib -lOpenCL -o test_global

test_simple_ocl: test_simple_ocl.cpp
	g++ -g test_simple_ocl.cpp -L/opt/rocm/lib -lOpenCL -o test_simple_ocl

clean:
	rm test test_global test_simple_ocl