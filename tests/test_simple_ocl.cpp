#include "../../ocl_simple/simple_ocl.hpp"

int main()
{
	init_ocl_env("./kernels.cl", {"assign_A_to_B"});
	int A = 1, B = 2;
	add_buf("A", sizeof(int), 1, &A);
	add_buf("B", sizeof(int), 1);
	run_kern("assign_A_to_B", {1}, "A", "B");
	read_buf(&B, "B");
	printf("A is %d, B is %d\n", A, B);
	return 0;
}