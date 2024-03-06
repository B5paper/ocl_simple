default: test test_global

rebuild:
	cd .. && make
	make

test: test.cpp
	g++ -g test.cpp -I.. -lOpenCL -o test

test_global: test_global.cpp
	g++ -g test_global.cpp -I.. -L.. -lglobal_ocl_env -lOpenCL -o test_global