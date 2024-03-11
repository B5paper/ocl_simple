#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <iostream>
using namespace std;

class OclBuf
{
	public:
    OclBuf(string name, size_t elm_size, size_t elm_num, cl_context context) {
        this->name = name;
        this->elm_size = elm_size;
        this->elm_num = elm_num;
        this->size = elm_size * elm_num;
		this->ctx = context;
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

	private:
	cl_context ctx;
};

class OclKern
{
    public:
    OclKern(cl_program program, string kernel_name) {
        int ret;
        kernel = clCreateKernel(program, kernel_name.c_str(), &ret);
        if (ret != CL_SUCCESS)
        {
            cout << "fail to create kernel" << endl;
            exit(-1);
        }

		clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint), &num_args, NULL);
        clear_args();
    }

    template<typename T> void set_arg(int idx, T &arg) {
        cl_int ret;
        ret = clSetKernelArg(kernel, idx, sizeof(T), &arg);
        if (ret != CL_SUCCESS)
        {
            cout << "fail to set arg" << endl;
            exit(-1);
        }
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
	cl_uint num_args;
	vector<size_t> arg_sizes;
    int cur_arg_idx;
};

struct OclEnv
{
    cl_platform_id plat;
	vector<cl_platform_id> plats;
    cl_device_id dev;
	vector<cl_device_id> devs;
    cl_context ctx;
    cl_program prog;
	cl_command_queue cmd_que;
    unordered_map<string, OclBuf> bufs;
    unordered_map<string, OclKern> kerns;

    OclEnv(string program_path, vector<string> kernel_names)
	{
		cl_uint num_plats;
		cl_int ret;
		ret = clGetPlatformIDs(0, NULL, &num_plats);
		if (ret != CL_SUCCESS)
		{
			cout << "fail to get opencl platform" << endl;
			exit(-1);
		}
		plats.resize(num_plats);
		ret = clGetPlatformIDs(num_plats, plats.data(), &num_plats);
		if (ret != CL_SUCCESS)
		{
			cout << "fail to get opencl platforms" << endl;
			exit(-1);
		}
		plat = plats[0];  // 直接选择第 1 个

		cl_uint num_devs;
		ret = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devs);
		if (ret != CL_SUCCESS)
		{
			cout << "fail to get device ids" << endl;
			exit(-1);
		}
		devs.resize(num_devs);
		ret = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, num_devs, devs.data(), &num_devs);
		if (ret != CL_SUCCESS)
		{
			cout << "fail to get device ids" << endl;
			exit(-1);
		}
		dev = devs[0];  // select the first device

		string dev_name;
		size_t buf_size;
		clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, NULL, &buf_size);
		dev_name.resize(buf_size);
		clGetDeviceInfo(dev, CL_DEVICE_NAME, buf_size, (void*) dev_name.data(), &buf_size);
		cout << "opencl device name: " << dev_name << endl;

		ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &ret);
		if (ret != CL_SUCCESS)
		{
			cout << "fail to create context" << endl;
			exit(-1);
		}

		cmd_que = clCreateCommandQueueWithProperties(ctx, dev, NULL, &ret);
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
		while (f.good())
		{
			getline(f, line);
			program_src.append(line);
			program_src.push_back('\n');
		}
		
		const char *p_program_src = program_src.data();
		size_t program_size = program_src.size();
		prog = clCreateProgramWithSource(ctx, 1, &p_program_src, &program_size, &ret);
		if (ret != CL_SUCCESS)
		{
			printf("fail to create program with source code\n");
			exit(-1);
		}

		ret = clBuildProgram(prog, 1, &dev, NULL, NULL, NULL);
		if (ret != CL_SUCCESS)
		{
			printf("fail to build program, error code: %d\n", ret);
			printf("Logs: \n");
			size_t log_length;
			ret = clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_length, NULL, &log_length);
			string log_content;
			log_content.resize(log_length);
			ret = clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, log_length, (void*) log_content.data(), &log_length);
			cout << log_content << endl;
			exit(-1);
		}

		for (string &kernel_name: kernel_names)
		{
			kerns.emplace(
				piecewise_construct,
				forward_as_tuple(kernel_name),
				forward_as_tuple(prog, kernel_name)
			);
		}
    }

    void add_buf(string buf_name, int elm_size, int elm_num) {
        bufs.emplace(
            piecewise_construct,
            forward_as_tuple(buf_name),
            forward_as_tuple(buf_name, elm_size, elm_num, ctx)
        );
    }

    void add_buf(string buf_name, int elm_size, int elm_num, void *src) {
        bufs.emplace(
            piecewise_construct,
            forward_as_tuple(buf_name),
            forward_as_tuple(buf_name, elm_size, elm_num, ctx)
        );

        write_buf(buf_name, src);
    }

    void del_buf(string buf_name) {
        bufs.erase(buf_name);
    }

    void write_buf(OclBuf &buf, void *src)
    {
        int ret;
        ret = clEnqueueWriteBuffer(cmd_que, buf.buf, CL_TRUE, 0, buf.size, src, 0, NULL, NULL);
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
        ret = clEnqueueWriteBuffer(cmd_que, buf.buf, CL_TRUE, off_elm_num * buf.elm_size, write_elm_num * buf.elm_size, src, 0, NULL, NULL);
        if (ret != CL_SUCCESS)
        {
            cout << "fail to write buffer" << endl;
            exit(-1);
        }
    }

    void read_buf(void *dst, OclBuf &buf)
    {
        int ret;
        ret = clEnqueueReadBuffer(cmd_que, buf.buf, CL_TRUE, 0, buf.size, dst, 0, NULL, NULL);
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
    void run_kernel(string kern_name, vector<size_t> global_work_size)
	{
		OclKern &kern = kerns.at(kern_name);
		kern.nd_range(global_work_size, cmd_que);
	}

    template<typename... Args>
    void run_kernel(string kern_name, vector<size_t> global_work_size, Args&&...args)
	{
		OclKern &kern = kerns.at(kern_name);
		kern.set_args(args...);
		kern.nd_range(global_work_size, cmd_que);
	}

    ~OclEnv() {
        cout << "[Warning] destroy ocl env" << endl;
    }
};


/*
	global part
*/
unique_ptr<OclEnv> global_ocl_env;
void init_ocl_env(string program_path, vector<string> kernel_names)
{
	global_ocl_env = make_unique<OclEnv>(program_path, kernel_names);
}

void add_buf(string buf_name, size_t elm_size, size_t elm_num)
{
	global_ocl_env->add_buf(buf_name, elm_size, elm_num);	
}

void add_buf(string buf_name, size_t elm_size, size_t elm_num, void *src)
{
	global_ocl_env->add_buf(buf_name, elm_size, elm_num, src);
}

void del_buf(string buf_name)
{
	global_ocl_env->del_buf(buf_name);
}

void write_buf(string buf_name, void *src)
{
	global_ocl_env->write_buf(buf_name, src);
}

void read_buf(void *src, string buf_name)
{
	global_ocl_env->read_buf(src, buf_name);
}

// 因为现在是 global 环境了，所以默认字符串都对应到 cl_mem 上，不再使用原本的类型了
// 感觉只要进了 OclEnv 这个环境里，都要按照这个规则来，以后有时间了把 OclEnv 中的 run_kernel() 也改一下吧
template<typename T>
void _set_args(OclKern &kern, T arg)
{
    if (typeid(arg) == typeid(const char*) ||
		typeid(arg) == typeid(char*) ||
		typeid(arg) == typeid(string))
	{
		OclBuf &buf = global_ocl_env->bufs.at(arg);
        kern.sa(buf);
	}
    else
    {
        kern.sa(arg);
    }
}

template<typename T, typename...Args>
void _set_args(OclKern &kern, T arg, Args...args)
{
    if (typeid(arg) == typeid(const char*) ||
		typeid(arg) == typeid(char*) ||
		typeid(arg) == typeid(string))
	{
		OclBuf &buf = global_ocl_env->bufs.at(arg);
        kern.sa(buf);
	}
    else
    {
        kern.sa(arg);
    }
    _set_args(kern, args...);
}

// 这个函数主要用于多加一行 clear_args()
template<typename...Args>
void _set_kernel_args(string kernel_name, Args...args)
{
    OclKern &kern = global_ocl_env->kerns.at(kernel_name);
    kern.clear_args();
    _set_args(kern, args...);
}

void run_kern(string kernel_name, vector<size_t> global_work_size)
{
    OclKern &kern = global_ocl_env->kerns.at(kernel_name);
    kern.nd_range(global_work_size, global_ocl_env->cmd_que);
}

template<typename...Args>
void run_kern(string kernel_name, vector<size_t> global_work_size, Args...args)
{
    _set_kernel_args(kernel_name, args...);
    OclKern &kern = global_ocl_env->kerns.at(kernel_name);
    kern.nd_range(global_work_size, global_ocl_env->cmd_que);
}


