#ifndef OCL_SIMPLE_H
#define OCL_SIMPLE_H

#define CL_TARGET_OPENCL_VERSION 300

#include <iostream>
#include <CL/cl.h>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
using namespace std;

struct OclBuf
{
    OclBuf(const char *name, cl_context context, size_t elm_size, size_t elm_num) {
        this->name = name;
        this->elm_size = elm_size;
        this->elm_num = elm_num;
        this->size = elm_size * elm_num;
        int ret;
        buf = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &ret);
        if (ret != CL_SUCCESS)
        {
            cout << "fail to create buffer" << endl;
            exit(-1);
        }
    }

    ~OclBuf() {
        int ret;
        ret = clReleaseMemObject(this->buf);
        if (ret != CL_SUCCESS)
        {
            cout << "fail to release buffer" << endl;
            exit(-1);
        }
        cout << "release ocl buffer: " << name << endl;
    }

    public:
    string name;
    int elm_size;
    int elm_num;
    int size;
    cl_mem buf;
};

class OclKern
{
    public:
    OclKern(cl_program program, const char *kernel_name) {
        int ret;
        kernel = clCreateKernel(program, kernel_name, &ret);
        if (ret != CL_SUCCESS)
        {
            cout << "fail to create kernel" << endl;
            exit(-1);
        }
        clear_args();

        // 未来可以根据这个函数得到的信息，推断 sa() 和 set_arg() 中，参数的类型和大小是否正确
        // ret = clGetKernelArgInfo();
    }

    template<typename T> OclKern& sa(T &arg) {
        int ret;
        ret = clSetKernelArg(kernel, cur_arg_idx, sizeof(T), &arg);
        cur_arg_idx += 1;
        if (ret != CL_SUCCESS)
        {
            cout << "fail to set kernel arg" << endl;
            exit(-1);
        }
        return *this;
    }

    // 不能在类的声明里进行模板的偏特化，所以这里直接选择放弃偏特化，重载了函数
    OclKern& sa(OclBuf &arg) {
        int ret;
        ret = clSetKernelArg(kernel, cur_arg_idx, sizeof(cl_mem), &arg.buf);  // 之所以要重载函数，是因为对于 OclBuf 类型，这里需要 .buf
        cur_arg_idx += 1;
        if (ret != CL_SUCCESS)
        {
            cout << "fail to set kernel arg" << endl;
            exit(-1);
        }
        return *this;
    }

    void clear_args() {
        cur_arg_idx = 0;
    }

    template<typename T> void set_arg(int idx, T &arg) {
        int ret;
        ret = clSetKernelArg(kernel, idx, sizeof(T), &arg);
        if (ret != CL_SUCCESS)
        {
            cout << "fail to set arg" << endl;
            exit(-1);
        }
    }

    template<typename T> void _set_args(T &t) {
        sa(t);
    }

    template<typename T, typename... Args> void _set_args(T &t, Args&&... args) {
        sa(t);
        _set_args(args...);
    }

    // 注意，Args 后必须加上 &&，其所依赖的模板函数参数也必须加上 &&，不然会按值传递参数，并调用析构函数，
    // 调用析构函数有可能会释放 opencl 申请的显存 buffer，整个程序会崩掉
    template<typename... Args> void set_args(Args&&... args) {
        clear_args();
        _set_args(args...);
    }

    void nd_range(vector<size_t> global_work_size, cl_command_queue command_queue) {
        uint work_dim = global_work_size.size();
        int ret;
        ret = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size.data(), NULL, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
        {
            cout << "fail to run kernel" << endl;
            exit(-1);
        }
    }

    public:
    string name;
    cl_kernel kernel;
    int cur_arg_idx;
};

struct OclEnv;
void init_ocl(OclEnv &ocl_env, string program_path, vector<string> program_names);

struct OclEnv
{
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_context context;
    cl_program program;
    cl_command_queue command_queue;
    unordered_map<string, OclBuf> bufs;
    unordered_map<string, OclKern> kerns;

    OclEnv(string program_path, vector<string> kernel_names) {
        init_ocl(*this, program_path, kernel_names);
    }

    void add_buf(string buf_name, int elm_size, int elm_num) {
        bufs.emplace(
            piecewise_construct,
            forward_as_tuple(buf_name),
            forward_as_tuple(buf_name.c_str(), context, elm_size, elm_num)
        );
    }

    void add_buf(string buf_name, int elm_size, int elm_num, void *src) {
        bufs.emplace(
            piecewise_construct,
            forward_as_tuple(buf_name),
            forward_as_tuple(buf_name.c_str(), context, elm_size, elm_num)
        );

        write_buf(buf_name, src);
    }

    void del_buf(string buf_name) {
        bufs.erase(buf_name);
    }

    void write_buf(OclBuf &buf, void *src)
    {
        int ret;
        ret = clEnqueueWriteBuffer(command_queue, buf.buf, CL_TRUE, 0, buf.size, src, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
        {
            cout << "fail to write buffer" << endl;
            exit(-1);
        }
    }

    void write_buf(string buf_name, void *src)
    {
        OclBuf &buf = bufs.at(buf_name);
        write_buf(buf, src);
    }

    void write_buf(string buf_name, void *src, size_t off_elm_num, size_t write_elm_num)
    {
        OclBuf &buf = bufs.at(buf_name);
        int ret;
        ret = clEnqueueWriteBuffer(command_queue, buf.buf, CL_TRUE, off_elm_num * buf.elm_size, write_elm_num * buf.elm_size, src, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
        {
            cout << "fail to write buffer" << endl;
            exit(-1);
        }
    }

    void read_buf(void *dst, OclBuf &buf)
    {
        int ret;
        ret = clEnqueueReadBuffer(command_queue, buf.buf, CL_TRUE, 0, buf.size, dst, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
        {
            cout << "fail to read buffer" << endl;
            exit(-1);
        }
    }

    void read_buf(void *dst, string buf_name)
    {
        OclBuf &buf = bufs.at(buf_name);
        read_buf(dst, buf);
    }

    template<typename... Args>
    void run_kernel(string kern_name, vector<size_t> global_work_size);

    template<typename... Args>
    void run_kernel(string kern_name, vector<size_t> global_work_size, Args&&...args);

    ~OclEnv() {
        cout << "[Warning] destroy ocl env" << endl;
    }
};

void init_ocl(OclEnv &ocl_env, string program_path, vector<string> kernel_names)
{
    unsigned int num_platforms_ids;
    int ret;
    ret = clGetPlatformIDs(1, &ocl_env.platform_id, &num_platforms_ids);
    if (ret != CL_SUCCESS)
    {
        cout << "fail to get opencl platform" << endl;
        exit(-1);
    }

    unsigned int num_device_ids;
    ret = clGetDeviceIDs(ocl_env.platform_id, CL_DEVICE_TYPE_GPU, 1, &ocl_env.device_id, &num_device_ids);
    if (ret != CL_SUCCESS)
    {
        cout << "fail to get device ids" << endl;
        exit(-1);
    }

    const int buf_size = 32;
    char buf[buf_size] = {0};
    clGetDeviceInfo(ocl_env.device_id, CL_DEVICE_NAME, buf_size, buf, NULL);
    cout << "opencl device name: " << buf << endl;

    ocl_env.context = clCreateContext(NULL, 1, &ocl_env.device_id, NULL, NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        cout << "fail to create context" << endl;
        exit(-1);
    }

    ocl_env.command_queue = clCreateCommandQueueWithProperties(ocl_env.context, ocl_env.device_id, NULL, &ret);
    if (ret != CL_SUCCESS)
    {
        cout << "fail to create command queue" << endl;
        exit(-1);
    }

    string program_src, line;
    ifstream f(program_path);
    if (!f.is_open())
    {
        cout << "fail to open opencl program file" << endl;
        exit(-1);
    }
    while (getline(f, line))
    {
        program_src.append(line);
        program_src.push_back('\n');
    }
    
    ocl_env.program = clCreateProgramWithSource(
        ocl_env.context, 1,
        (const char **)&static_cast<const char* const &>(program_src.c_str()),
        &static_cast<const size_t&>(program_src.size()),
        &ret
    );
    if (ret != CL_SUCCESS)
    {
        printf("fail to create program with source code\n");
        exit(-1);
    }

    ret = clBuildProgram(ocl_env.program, 1, &ocl_env.device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        printf("fail to build program, error code: %d\n", ret);
        printf("Logs: \n");
        size_t max_log_length = 1024 * 10;
        char *build_log = (char*) malloc(max_log_length);
        size_t actual_log_length;
        ret = clGetProgramBuildInfo(ocl_env.program, ocl_env.device_id, CL_PROGRAM_BUILD_LOG, max_log_length, build_log, &actual_log_length);
        printf("%s\n", build_log);
        free(build_log);
        exit(-1);
    }

    for (string &kernel_name: kernel_names)
    {
        ocl_env.kerns.emplace(
            piecewise_construct,
            forward_as_tuple(kernel_name),
            forward_as_tuple(ocl_env.program, kernel_name.c_str())
        );
    }
}

void read_buf(void *dst, OclBuf &buf, OclEnv &env)
{
    int ret;
    ret = clEnqueueReadBuffer(env.command_queue, buf.buf, CL_TRUE, 0, buf.size, dst, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        cout << "fail to read buffer" << endl;
        exit(-1);
    }
}

void read_buf(void *dst, string buf_name, OclEnv &ocl_env)
{
    OclBuf &buf = ocl_env.bufs.at(buf_name);
    read_buf(dst, buf, ocl_env);
}

void write_buf(OclBuf &buf, void *src, OclEnv &env)
{
    int ret;
    ret = clEnqueueWriteBuffer(env.command_queue, buf.buf, CL_TRUE, 0, buf.size, src, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        cout << "fail to write buffer" << endl;
        exit(-1);
    }
}

void write_buf(string buf_name, void *src, OclEnv &ocl_env)
{
    OclBuf &buf = ocl_env.bufs.at(buf_name);
    write_buf(buf, src, ocl_env);
}

void env_add_buf(OclEnv &ocl_env, string buf_name, int elm_size, int elm_num)
{
    ocl_env.bufs.emplace(
        piecewise_construct,
        forward_as_tuple(buf_name),
        forward_as_tuple(buf_name.c_str(), ocl_env.context, elm_size, elm_num)
    );
}

void set_buf_as_kern_arg(OclKern &k, vector<string> buf_names, OclEnv &env)
{
    k.clear_args();
    for (int i = 0; i < buf_names.size(); ++i)
    {
        string &buf_name = buf_names[i];
        k.sa(env.bufs.at(buf_name).buf);
    }
}

void _run_kern(OclKern &k, vector<size_t> global_work_size, OclEnv &ocl_env)
{
    uint work_dim = global_work_size.size();
    int ret;
    ret = clEnqueueNDRangeKernel(ocl_env.command_queue, k.kernel, work_dim, NULL, global_work_size.data(), NULL, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
    {
        cout << "fail to run kernel" << endl;
        exit(-1);
    }
}

void set_arg_and_run(OclKern &k, vector<string> buf_names, vector<size_t> global_work_size, OclEnv &env)
{
    set_buf_as_kern_arg(k, buf_names, env);
    _run_kern(k, global_work_size, env);
}

void set_arg_and_run(string kern_name, vector<string> buf_names, vector<size_t> global_work_size, OclEnv &env)
{
    OclKern &k = env.kerns.at(kern_name);
    set_arg_and_run(k, buf_names, global_work_size, env);
}

void env_del_buf(OclEnv &ocl_env, string buf_name)
{
    ocl_env.bufs.erase(buf_name);
}

template<typename T>
void _set_args_with_env(OclKern &k, OclEnv &ocl_env, T &&t)
{
    k.sa(t);
}

void _set_args_with_env(OclKern &k, OclEnv &ocl_env, const char *buf_name)
{
    OclBuf &buf = ocl_env.bufs.at(buf_name);
    k.sa(buf);
}

template<typename T, typename...Args>
void _set_args_with_env(OclKern &k, OclEnv &ocl_env, T &&t, Args&&...args)
{
    k.sa(t);
    _set_args_with_env(k, ocl_env, args...);
}

template<typename...Args>
// 这里如果写成 string && 会运行时报错
// 不清楚为什么
// 我希望能处理 sring, char*, const char*，string&, string&&，const string，const string&，const string&& 这些类型
void _set_args_with_env(OclKern &k, OclEnv &ocl_env, const char *buf_name, Args&&...args)
{
    OclBuf &buf = ocl_env.bufs.at(buf_name);
    k.sa(buf);
    _set_args_with_env(k, ocl_env, args...);
}

template<typename... Args>
void set_args_with_env(OclKern &k, OclEnv &ocl_env, Args&&...args)
{
    k.clear_args();
    _set_args_with_env(k, ocl_env, args...);
}

template<typename... Args>
void set_args_with_env(string kern_name, OclEnv &ocl_env, Args&&...args)
{
    OclKern &k = ocl_env.kerns.at(kern_name);
    k.clear_args();
    _set_args_with_env(k, ocl_env, args...);
}

template<typename... Args>
void run_kernel(string kern_name, OclEnv &ocl_env, vector<size_t> global_work_size, Args&&...args)
{
    OclKern &k = ocl_env.kerns.at(kern_name);
    set_args_with_env(k, ocl_env, args...);
    _run_kern(k, global_work_size, ocl_env);
}

template<typename... Args>
void OclEnv::run_kernel(string kern_name, vector<size_t> global_work_size)
{
    OclKern &k = (*this).kerns.at(kern_name);
    // set_args_with_env(k, *this);
    _run_kern(k, global_work_size, *this);
}

template<typename... Args>
void OclEnv::run_kernel(string kern_name, vector<size_t> global_work_size, Args&&...args)
{
    OclKern &k = (*this).kerns.at(kern_name);
    set_args_with_env(k, *this, args...);
    _run_kern(k, global_work_size, *this);
}

ostream& operator<<(ostream &cout, cl_float3 &vec)
{
    cout << "[";
    for (int i = 0; i < 3; ++i)
        cout << vec.s[i] << ", ";
    cout << vec.s[3] << "]";
    return cout;
}

// ostream& operator<<(ostream &cout, cl_float4 &vec)
// {
//     cout << "[";
//     for (int i = 0; i < 3; ++i)
//         cout << vec.s[i] << ", ";
//     cout << vec.s[3] << "]";
//     return cout;
// }


#endif