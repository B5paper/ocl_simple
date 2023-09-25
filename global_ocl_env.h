#ifndef GLOBAL_OCL_ENV_H
#define GLOBAL_OCL_ENV_H

#include "ocl_simple.h"
#include <memory>

static unique_ptr<OclEnv> pocl_env;  // 自动释放内存，因为 OclEnv 没有默认构造函数，所以必须做成指针的形式

void init_global_ocl_env(string program_path, vector<string> kernel_names)
{
    pocl_env = make_unique<OclEnv>(program_path, kernel_names);
}

void add_buf(string buf_name, int elm_size, int elm_num)
{
    pocl_env->add_buf(buf_name, elm_size, elm_num);
}

void add_buf(string buf_name, int elm_size, int elm_num, void *src)
{
    pocl_env->add_buf(buf_name, elm_size, elm_num, src);
}

void del_buf(string buf_name)
{
    pocl_env->del_buf(buf_name);
}

void read_buf(void *dst, string buf_name)
{
    pocl_env->read_buf(dst, buf_name);
}

void write_buf(string buf_name, void *src)
{
    pocl_env->write_buf(buf_name, src);
}

template<typename ...Args>
void run_kern(string kernel_name, vector<size_t> work_sizes, Args ...args)
{
    pocl_env->run_kernel(kernel_name, work_sizes, args...);
}

#endif