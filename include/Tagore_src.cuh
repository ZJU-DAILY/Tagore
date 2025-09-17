#include <stdio.h>
#include<chrono>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <set>

#include <curand_kernel.h>
#include <random>
#include <cuda_fp16.h>
#include <cstring>

#include <cstddef>
#include <mutex>
#include <bitset>
#include <cmath>
#include <boost/dynamic_bitset.hpp>
#include <stack>


// fixed
#define RESERVENUM 128
#define SS 128
#define BLK_H 16
#define BLK_W 16
#define MAX_P 512
#define SAMPLE 32
#define INF_DIS 10000000
#define SELECT_CAND 480

// K
#define K_SIZE 96
// DIM
#define DIM_SIZE 128
// TOPM
#define TOPM_SIZE 64
// FINAL_DEGREE 32
#define FINAL_DEGREE_SIZE 64

using namespace nvcuda;

using namespace std;

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

__device__ void swap(float &a, float &b);

__device__ void swap_ids(unsigned &a, unsigned &b);

__device__ void swap_bool(bool &a, bool &b);

__device__ void bitonic_sort_new2(float* shared_arr, unsigned* ids, unsigned* ids2, unsigned len);

__device__ void bitonic_sort_id_new2(unsigned* shared_arr, unsigned len);

__device__ void bitonic_sort_id_by_dis(float* shared_arr, unsigned* ids, bool* visit, unsigned len);

__device__ void bitonic_sort_id_and_dis(float* shared_arr, unsigned* ids, unsigned len);

__device__ void bitonic_sort_by_id(float* shared_arr, unsigned* ids, unsigned len);

__device__ void bitonic_sort_id_by_detour(unsigned* shared_arr, unsigned* ids, unsigned len);

__device__ inline unsigned cinn_nvgpu_uniform_random(unsigned long long seed, unsigned node_num);

struct __align__(8) half4 {
    half2 x, y;
};

__device__ __forceinline__ half4 BitCast(const float2& src) noexcept {
    half4 dst;
    std::memcpy(&dst, &src, sizeof(half4));
    return dst;
}

__device__ __forceinline__ half4 Load(const half* address) {
    float2 x = __ldg(reinterpret_cast<const float2*>(address));
    return BitCast(x);
}

__device__ __forceinline__ half2 BitCast_half2(const float& src) noexcept {
    half2 dst;
    std::memcpy(&dst, &src, sizeof(half2));
    return dst;
}

__device__ __forceinline__ half2 Load_half2(const half* address) {
    float x = __ldg(reinterpret_cast<const float*>(address));
    return BitCast_half2(x);
}

__global__ void initialize_graph(unsigned* graph, unsigned node_num, float* nei_distance, bool* nei_visit, unsigned K);

__global__ void sample_kernel6(unsigned* graph, unsigned* reverse_graph, unsigned it, const half* __restrict__ values, unsigned node_num, unsigned all_it, float* nei_distance, float* reverse_distance, bool* nei_visit, unsigned* reverse_num, unsigned DIM, unsigned K);

__global__ void reset_reverse_num(unsigned* reverse_num, unsigned POINTS);

__global__ void merge_reverse_plus(unsigned* graph, unsigned* reverse_graph, float* nei_distance, float* reverse_distance, unsigned* reverse_num, unsigned K);

__global__ void reset_reverse_new_old_num(unsigned* reverse_num, unsigned* reverse_num_old, unsigned POINTS);

__global__ void nn_descent_opt_sample(unsigned* graph, unsigned* reverse_graph, bool* nei_visit, unsigned* reverse_num, unsigned* reverse_num_old,unsigned* new_num_global,unsigned* old_num_global, unsigned* hybrid_list, unsigned K);

__global__ void nn_descent_opt_reverse_sample(unsigned* graph, unsigned* reverse_graph, unsigned* reverse_num, unsigned* reverse_num_old, unsigned* new_num_global,unsigned* old_num_global,unsigned* hybrid_list,unsigned K);

__global__ void nn_descent_opt_cal(unsigned* graph, unsigned* reverse_graph, half* data, float* data_power, unsigned it, float* reverse_distance, unsigned* reverse_num, unsigned* new_num_global, unsigned* old_num_global, unsigned* hybrid_list, unsigned DIM, unsigned K);

__global__ void nn_descent_opt_merge(unsigned* graph, unsigned* reverse_graph, unsigned it, unsigned all_it, float* nei_distance, float* reverse_distance, bool* nei_visit, unsigned* reverse_num, unsigned K);

__global__ void cal_power(half* data_half, float* data_power, unsigned dim, unsigned DIM);

__global__ void do_reverse_graph(unsigned* graph_dev, unsigned* reverse_graph_dev, float* nei_distance, float* reverse_distance, unsigned* reverse_num, unsigned K);

__global__ void reset_visit_reversenum(unsigned* reverse_graph, bool* nei_visit, unsigned* reverse_num, unsigned POINTS, unsigned K);

typedef float(*funcFormat)(float, float, float, float);

__device__ float distance_filter(float res, float dis, float cur_dis, float threshold);

__device__ float angle_filter(float res, float dis, float cur_dis, float threshold);

__global__ void select_path(unsigned* graph, unsigned* reverse_graph, unsigned *ep, const half* __restrict__ values, unsigned max_cand, float* nei_distance, float* reverse_distance, unsigned* reverse_num, unsigned DIM, unsigned FINAL_DEGREE, unsigned TOPM, unsigned K, float thre);

__global__ void select_2hop(unsigned* graph, unsigned* reverse_graph, const half* __restrict__ values, unsigned max_cand, float* nei_distance, float* reverse_distance, unsigned* reverse_num, unsigned DIM, unsigned FINAL_DEGREE, unsigned TOPM, unsigned K, float thre);

__global__ void filter_reverse(unsigned* graph, unsigned* reverse_graph, half* values, float* nei_distance, float* reverse_distance, unsigned* reverse_num, unsigned DIM, unsigned FINAL_DEGREE, unsigned K, funcFormat opert, float thre);

__global__ void cal_ep_gpu(unsigned* graph, unsigned* reverse_graph, unsigned* ep, const half* __restrict__ centers, const half* __restrict__ values, unsigned max_cand, unsigned DIM, unsigned TOPM, unsigned K);

__global__ void select_1hop_cagra(unsigned* graph, unsigned d, unsigned* reverse_graph, unsigned K, unsigned* reverse_num, unsigned* new_list);

__global__ void filter_reverse_1hop(unsigned* graph, unsigned* reverse_graph, unsigned d, unsigned K, unsigned* reverse_num, unsigned* new_list);

__global__ void select_1hop_dpg(unsigned* graph, unsigned d, unsigned* reverse_graph, unsigned K, unsigned* reverse_num, unsigned* new_list, unsigned DIM, const half* __restrict__ values, float* nei_distance);
