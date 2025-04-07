#include <Tagore.cuh>

using namespace nvcuda;

using namespace std;

__device__ void swap(float &a, float &b){
    float t = a;
    a = b;
    b = t;
}

__device__ void swap_ids(unsigned &a, unsigned &b){
    unsigned t = a;
    a = b;
    b = t;
}

__device__ void swap_bool(bool &a, bool &b){
    bool t = a;
    a = b;
    b = t;
}

__device__ void bitonic_sort_new2(float* shared_arr, unsigned* ids, unsigned* ids2, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    // if(tid < len / 2){
        for(unsigned stride = 1; stride < len; stride <<= 1){
            for(unsigned step = stride; step > 0; step >>= 1){
                for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                    unsigned a = 2 * step * (k / step);
                    unsigned b = k % step;
                    unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                    unsigned d = a + b + step;
                    if(d < len && shared_arr[u] > shared_arr[d]){
                        swap(shared_arr[u],shared_arr[d]);
                        swap_ids(ids[u],ids[d]);
                        swap_ids(ids2[u],ids2[d]);
                    }
                }
                __syncthreads();
            }
        }
    // }
    // __syncthreads();
}

__device__ void bitonic_sort_id_new2(unsigned* shared_arr, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
        for(unsigned stride = 1; stride < len; stride <<= 1){
            for(unsigned step = stride; step > 0; step >>= 1){
                for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                    unsigned a = 2 * step * (k / step);
                    unsigned b = k % step;
                    unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                    unsigned d = a + b + step;
                    if(d < len && shared_arr[u] > shared_arr[d]){
                        swap_ids(shared_arr[u],shared_arr[d]);
                    }
                }
                __syncthreads();
            }
        }
}

__device__ void bitonic_sort_id_by_dis(float* shared_arr, unsigned* ids, bool* visit, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                unsigned a = 2 * step * (k / step);
                unsigned b = k % step;
                unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                unsigned d = a + b + step;
                if(d < len && shared_arr[u] > shared_arr[d]){
                    swap(shared_arr[u],shared_arr[d]);
                    swap_ids(ids[u],ids[d]);
                    swap_bool(visit[u], visit[d]);
                }
            }
            __syncthreads();
        }
    }
}

__device__ void bitonic_sort_id_and_dis(float* shared_arr, unsigned* ids, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                unsigned a = 2 * step * (k / step);
                unsigned b = k % step;
                unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                unsigned d = a + b + step;
                if(d < len && shared_arr[u] > shared_arr[d]){
                    swap(shared_arr[u],shared_arr[d]);
                    swap_ids(ids[u],ids[d]);
                }
            }
            __syncthreads();
        }
    }
}

__device__ void bitonic_sort_by_id(float* shared_arr, unsigned* ids, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                unsigned a = 2 * step * (k / step);
                unsigned b = k % step;
                unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                unsigned d = a + b + step;
                if(d < len && ids[u] > ids[d]){
                    swap(shared_arr[u],shared_arr[d]);
                    swap_ids(ids[u],ids[d]);
                }
            }
            __syncthreads();
        }
    }
}

__device__ void bitonic_sort_id_by_detour(unsigned* shared_arr, unsigned* ids, unsigned len){
    const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
    for(unsigned stride = 1; stride < len; stride <<= 1){
        for(unsigned step = stride; step > 0; step >>= 1){
            for(unsigned k = tid; k < len / 2; k += blockDim.x * blockDim.y){
                unsigned a = 2 * step * (k / step);
                unsigned b = k % step;
                unsigned u = ((step == stride) ? (a + step - 1 - b) : (a + b));
                unsigned d = a + b + step;
                if(d < len && shared_arr[u] > shared_arr[d]){
                    swap_ids(shared_arr[u],shared_arr[d]);
                    swap_ids(ids[u],ids[d]);
                }
            }
            __syncthreads();
        }
    }
}

__device__ inline unsigned cinn_nvgpu_uniform_random(unsigned long long seed, unsigned node_num){
    curandStatePhilox4_32_10_t state;
    int idx = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    curand_init(seed, idx, 1, &state);
    return (unsigned)(((float)node_num) * curand_uniform(&state)) % node_num;
}

__global__ void initialize_graph(unsigned* graph, unsigned node_num, float* nei_distance, bool* nei_visit, unsigned K){
    unsigned bid = blockIdx.x, laneid = threadIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x; 
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        graph[bid * K + i] = cinn_nvgpu_uniform_random(clock()*(unsigned long long)bid, node_num); // (K = 32 * n, is this assumption correct?)

        nei_distance[bid * K + i] = INF_DIS;
        nei_visit[bid*K+i] = false;
    }
}

__global__ void sample_kernel6(unsigned* graph, unsigned* reverse_graph, unsigned it, const half* __restrict__ values, unsigned node_num, unsigned all_it, float* nei_distance, float* reverse_distance, bool* nei_visit, unsigned* reverse_num, unsigned DIM, unsigned K){
    unsigned tmp_it = threadIdx.y/4;
    unsigned bid = blockIdx.x, laneid = threadIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ unsigned fetch_ids[4];
    __shared__ int lock;
    __shared__ unsigned new_list_shared[MAX_P], nei_list_shared[K_SIZE], new_list2[MAX_P];
    __shared__ float new_dist_shared[MAX_P], nei_dist_shared[K_SIZE], new_dist2[MAX_P];
    __shared__ bool nei_visit_shared[K_SIZE];
    __shared__ half4 tmp_val_sha[DIM_SIZE / 4];
    if(tid == 0) lock = 1;
    __syncthreads();
    if(threadIdx.y % 4 == 0 && laneid == 0){
        while (atomicExch(&lock, 0) == 0);
        while(nei_visit[bid*K+tmp_it] == true && tmp_it < K){
            tmp_it++;
        }
        nei_visit[bid*K+tmp_it] = true;
        lock = 1;
        fetch_ids[threadIdx.y / 4] = tmp_it % K;
    }
    __syncthreads();
    tmp_it = fetch_ids[threadIdx.y / 4];

    unsigned samp_id = graph[bid * K + tmp_it];
    if(threadIdx.y % 4 < 2){
        new_list_shared[tid] = graph[samp_id * K + laneid + (threadIdx.y%4)*32];
        // if(new_list_shared[tid] == 0xFFFFFFFF) printf("A");
    }
    else{
        new_list_shared[tid] = reverse_graph[samp_id * K + cinn_nvgpu_uniform_random(clock(), K)];
        // if(new_list_shared[tid] == 0xFFFFFFFF) printf("B");
    }

    // sort (and remove duplicates)

    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        new_list2[i] = graph[bid * K + i];
    }
    __syncthreads();

    bitonic_sort_id_new2(new_list_shared, MAX_P);
    bitonic_sort_id_new2(new_list2, K);

    for(unsigned i = tid + 1; i < MAX_P; i += blockDim.x * blockDim.y){
        new_dist_shared[i] = (new_list_shared[i] == new_list_shared[i - 1] ? INF_DIS : (new_list_shared[i] == bid ? INF_DIS : 0));
    }
    // __syncthreads();
    if(tid == 0){
        if(new_list_shared[0] != bid){
            new_dist_shared[0] = 0;
        }
        else{
            new_dist_shared[0] = INF_DIS;
        }
    }
    __syncthreads();
    for(unsigned i = tid; i < MAX_P; i += blockDim.x * blockDim.y){
        if(new_dist_shared[i] > 1.0f) continue;
        unsigned tmp = K, res_id = 0;
        unsigned val = new_list_shared[i];
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            unsigned cand = new_list2[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (new_list2[res_id] < val);
        unsigned tmp_count = 0;
        if(new_list2[res_id] == new_list_shared[i]){
            new_dist_shared[i] = INF_DIS;
        }
    }
    __syncthreads();

    // half4 val1;
    // if(laneid < 24) val1 = Load(&values[bid * DIM + 4 * laneid]);
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&values[bid * DIM + 4 * i]);
	}
	__syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < MAX_P; i += blockDim.y){
        if(new_dist_shared[i] < 1.0){
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;

            // if(laneid < 24){
            //     half4 val2 = Load(&values[new_list_shared[i] * DIM + laneid * 4]);
                
            //     val_res = __hmul2(__hsub2(val1.x, val2.x), __hsub2(val1.x, val2.x));
            //     val_res = __hfma2(__hsub2(val1.y, val2.y), __hsub2(val1.y, val2.y), val_res);
            // }
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                half4 val2 = Load(&values[new_list_shared[i] * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }

            if(laneid == 0){
                new_dist_shared[i] = __half2float(__hadd(val_res.x, val_res.y));
            }
        }
    }
    __syncthreads();
    for(unsigned i = tid; i < MAX_P; i += blockDim.x * blockDim.y){
        new_list2[i] = new_list_shared[i];
        new_dist2[i] = new_dist_shared[i];
    }

    // merge
    // sort (and remove duplicates)
    
    float min_ele = nei_distance[bid * K + K - 1];
    __shared__ unsigned dup_num;
    if(tid == 0) dup_num = 0;
    __syncthreads();

    for(unsigned i = tid; i < MAX_P; i += blockDim.x * blockDim.y){
        if(new_dist2[i] < min_ele){
            unsigned id_tmp = atomicAdd(&dup_num, 1);
            new_list_shared[id_tmp] = new_list2[i];
            new_dist_shared[id_tmp] = new_dist2[i];
        }
    }
    
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        nei_visit_shared[i] = nei_visit[bid*K+i];
        nei_dist_shared[i] = nei_distance[bid*K+i];
        nei_list_shared[i] = graph[bid * K + i];
    }
    __syncthreads();
    
    if(bid == 0 && tid == 0) printf("Dup: %d %d\n", dup_num, it);

    for(unsigned i = tid; i < dup_num; i += blockDim.x * blockDim.y){
        float val = new_dist_shared[i];
        // binary search
        unsigned tmp = K, res_id = 0;
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            float cand = nei_dist_shared[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (nei_dist_shared[res_id] < val);
        unsigned tmp_count = 0;
        while(res_id + tmp_count < K && nei_dist_shared[res_id + tmp_count] == new_dist_shared[i]){
            if(nei_list_shared[res_id + tmp_count] == new_list_shared[i]){
                new_dist_shared[i] = INF_DIS;
                res_id = K;
                break;
            }
            tmp_count++;
        }
        new_list2[i] = res_id;
    }
    __syncthreads();

    
    bitonic_sort_new2(new_dist_shared, new_list_shared, new_list2, dup_num);

    // merge
    if(dup_num > 0){
    for(unsigned i = threadIdx.y; i < (K + dup_num + blockDim.x - 1) / blockDim.x; i += blockDim.y){
        unsigned res_id = 0, id_reg;
        bool visit_reg = false;
        float val;
        if(i < K / blockDim.x){
            val = nei_dist_shared[laneid + i * blockDim.x];
            id_reg = nei_list_shared[laneid + i * blockDim.x];
            visit_reg = nei_visit_shared[laneid + i * blockDim.x];
            unsigned tmp = dup_num;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = new_dist_shared[res_id + halfsize];
                res_id += ((cand <= val) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id += (new_dist_shared[res_id] <= val);
            res_id += (laneid + i * blockDim.x);
        }
        else{
            if(laneid + i * blockDim.x - K < dup_num){
                val = new_dist_shared[laneid + i * blockDim.x - K];
                id_reg = new_list_shared[laneid + i * blockDim.x - K];
                res_id = (new_list2[laneid + i * blockDim.x - K] + laneid + i * blockDim.x - K);
            }
            else{
                res_id = K;
            }
        }
        __syncthreads();
        if(res_id < K){
            nei_distance[bid * K + res_id] = val;
            graph[bid * K + res_id] = id_reg;
            nei_visit[bid*K+res_id] = visit_reg;
            unsigned tmp_id = atomicAdd(&reverse_num[id_reg], 1);
            if(tmp_id < K) {
                reverse_graph[id_reg * K + tmp_id] = bid;
                reverse_distance[id_reg*RESERVENUM+tmp_id] = val;
            }
            else{
                if(val < reverse_distance[id_reg*RESERVENUM+(tmp_id % K)]){
                    reverse_graph[id_reg * K + tmp_id % K] = bid;
                    reverse_distance[id_reg*RESERVENUM+(tmp_id % K)] = val;
                }
            }
        }
    }
    }
    __syncthreads();
    if(it == all_it - 1 && bid == 0 && tid == 0){
        for(unsigned i = 0; i < K; i++)
            printf("%f, ", nei_distance[bid * K + i]);
        printf("\n");
        for(unsigned i = 0; i < K; i++)
            printf("%d, ", graph[bid * K + i]);
        printf("\n");
    }
}

__global__ void reset_reverse_num(unsigned* reverse_num, unsigned POINTS){
    unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(unsigned i = tid ; i < POINTS; i += gridDim.x * blockDim.x){
        reverse_num[i] = 0;
    }
}

__global__ void merge_reverse_plus(unsigned* graph, unsigned* reverse_graph, float* nei_distance, float* reverse_distance, unsigned* reverse_num, unsigned K){
    unsigned bid = blockIdx.x, laneid = threadIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ unsigned graph_nei[K_SIZE], reverse_nei[K_SIZE], new_list2[K_SIZE];
    __shared__ float graph_nei_dist[K_SIZE], reverse_nei_dist[K_SIZE];
    unsigned reverse_act_num = min(K, reverse_num[bid]);
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        graph_nei[i] = graph[bid * K + i];
        graph_nei_dist[i] = nei_distance[bid*K+i];
    }
    for(unsigned i = tid; i < reverse_act_num; i += blockDim.x * blockDim.y){
        reverse_nei[i] = reverse_graph[bid * K + i];
        reverse_nei_dist[i] = reverse_distance[bid*RESERVENUM+i];
    }
    __syncthreads();

    bitonic_sort_by_id(reverse_nei_dist, reverse_nei, reverse_act_num);
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        unsigned val = graph_nei[i];
        // binary search
        unsigned tmp = reverse_act_num, res_id = 0;
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            unsigned cand = reverse_nei[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (reverse_nei[res_id] < val);
        if(reverse_nei[res_id] == graph_nei[i]){
            new_list2[res_id] = K;
            reverse_nei_dist[res_id] = INF_DIS;
        }
    }
    __syncthreads();
    for(unsigned i = tid; i < reverse_act_num; i += blockDim.x * blockDim.y){
        float val = reverse_nei_dist[i];
        // binary search
        unsigned tmp = K, res_id = 0;
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            float cand = graph_nei_dist[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (graph_nei_dist[res_id] < val);
        unsigned tmp_count = 0;
        while(res_id + tmp_count < K && abs(graph_nei_dist[res_id + tmp_count] - reverse_nei_dist[i]) < 0.1){
            if(graph_nei[res_id + tmp_count] == reverse_nei[i]){
                reverse_nei_dist[i] = INF_DIS;
                res_id = K;
                break;
            }
            tmp_count++;
        }
        new_list2[i] = res_id;
    }
    __syncthreads();
    bitonic_sort_new2(reverse_nei_dist, reverse_nei, new_list2, reverse_act_num);

    if(reverse_act_num > 0){
    for(unsigned i = threadIdx.y; i < (K + reverse_act_num + blockDim.x - 1) / blockDim.x; i += blockDim.y){
        unsigned res_id = 0, id_reg;
        float val;
        if(i < K / blockDim.x){
            val = graph_nei_dist[laneid + i * blockDim.x];
            id_reg = graph_nei[laneid + i * blockDim.x];
            unsigned tmp = reverse_act_num;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                float cand = reverse_nei_dist[res_id + halfsize];
                res_id += ((cand <= val) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id += (reverse_nei_dist[res_id] <= val);
            res_id += (laneid + i * blockDim.x);
        }
        else{
            if(laneid + i * blockDim.x - K < reverse_act_num){
                val = reverse_nei_dist[laneid + i * blockDim.x - K];
                id_reg = reverse_nei[laneid + i * blockDim.x - K];
                res_id = (new_list2[laneid + i * blockDim.x - K] + laneid + i * blockDim.x - K);
            }
            else{
                res_id = K;
            }
        }
        __syncthreads();
        if(res_id < K){
            nei_distance[bid*K+res_id] = val;
            graph[bid * K + res_id] = id_reg;
        }
    }
    }
}

__global__ void reset_reverse_new_old_num(unsigned* reverse_num, unsigned* reverse_num_old, unsigned POINTS){
    unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(unsigned i = tid ; i < POINTS; i += gridDim.x * blockDim.x){
        reverse_num[i] = 0;
        reverse_num_old[i] = 0;
    }
}

__global__ void nn_descent_opt_sample(unsigned* graph, unsigned* reverse_graph, bool* nei_visit, unsigned* reverse_num, unsigned* reverse_num_old,unsigned* new_num_global,unsigned* old_num_global, unsigned* hybrid_list, unsigned K){
    unsigned bid = blockIdx.x, laneid = threadIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ unsigned new_nn[SAMPLE], old_nn[SAMPLE];

    __shared__ unsigned new_nn_count, old_nn_count;
    if(tid == 0){
        new_nn_count = 0;
        old_nn_count = 0;
        unsigned pointer = 0;
        while(new_nn_count < SAMPLE && old_nn_count < SAMPLE && pointer < K){
            if(nei_visit[bid*K+pointer] == false){
                new_nn[new_nn_count] = graph[bid * K + pointer];
                nei_visit[bid*K+pointer] = true;
                new_nn_count++;
            }
            else{
                old_nn[old_nn_count] = graph[bid * K + pointer];
                old_nn_count++;
            }
            pointer++;
        }
        while(old_nn_count < SAMPLE && pointer < K){
            if(nei_visit[bid*K+pointer] == true){
                old_nn[old_nn_count] = graph[bid * K + pointer];
                old_nn_count++;
            }
            pointer++;
        }
        while(new_nn_count < SAMPLE && pointer < K){
            if(nei_visit[bid*K+pointer] == false){
                new_nn[new_nn_count] = graph[bid * K + pointer];
                nei_visit[bid*K+pointer] = true;
                new_nn_count++;
            }
            pointer++;
        }
    }
    __syncthreads();

    for(unsigned i = tid; i < new_nn_count; i += blockDim.x * blockDim.y){
        unsigned id_tmp = atomicAdd(&reverse_num[new_nn[i]], 1);
        if(id_tmp < SAMPLE) {
            reverse_graph[new_nn[i] * K + id_tmp] = bid;
        }
        // new_list[bid][1 + i] = new_nn[i];
        hybrid_list[bid*(SAMPLE * 4)+i] = new_nn[i];
    }
    for(unsigned i = tid; i < old_nn_count; i += blockDim.x * blockDim.y){
        unsigned id_tmp = atomicAdd(&reverse_num_old[old_nn[i]], 1);
        if(id_tmp < SAMPLE) {
            // reverse_graph_old[old_nn[i]][id_tmp] = bid;
            reverse_graph[old_nn[i] * K + SAMPLE + id_tmp] = bid;
        }
        // old_list[bid][1 + i] = old_nn[i];
        hybrid_list[bid*(SAMPLE * 4)+2 * SAMPLE + i] = old_nn[i];
    }
    if(tid == 0){
        // old_list[bid][0] = old_nn_count;
        // new_list[bid][0] = new_nn_count;
        new_num_global[bid] = new_nn_count;
        old_num_global[bid] = old_nn_count;
        if(bid == 0) printf("%d %d | ", old_nn_count, new_nn_count);
    }
}

__global__ void nn_descent_opt_reverse_sample(unsigned* graph, unsigned* reverse_graph, unsigned* reverse_num, unsigned* reverse_num_old, unsigned* new_num_global,unsigned* old_num_global,unsigned* hybrid_list,unsigned K){
    unsigned bid = blockIdx.x, laneid = threadIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ unsigned new_nn[SAMPLE], old_nn[SAMPLE];

    unsigned new_num = new_num_global[bid];
    for(unsigned i = tid; i < new_num; i += blockDim.x * blockDim.y){
        new_nn[i] = hybrid_list[bid*(SAMPLE * 4)+i];
    }
    unsigned new_reverse_num = min(reverse_num[bid], SAMPLE);

    // unsigned old_num = old_list[bid][0];
    unsigned old_num = old_num_global[bid];
    for(unsigned i = tid; i < old_num; i += blockDim.x * blockDim.y){
        // old_nn[i] = old_list[bid][1 + i];
        old_nn[i] = hybrid_list[bid*(SAMPLE * 4)+2 * SAMPLE + i];
    }
    unsigned old_reverse_num = min(reverse_num_old[bid], SAMPLE);

    __syncthreads();

    for(unsigned i = tid; i < new_reverse_num; i += blockDim.x * blockDim.y){
        // unsigned val = reverse_graph[bid * K + i];
        unsigned val = reverse_graph[bid * K + i];
        // binary search
        unsigned tmp = new_num, res_id = 0;
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            unsigned cand = new_nn[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (new_nn[res_id] < val);
        if(new_nn[res_id] == val){
            reverse_graph[bid * K + i] = 0xFFFFFFFF;
            // hybrid_list[bid][2 * SAMPLE + i] = 0xFFFFFFFF;
        }
        else{
            // unsigned id_tmp = atomicAdd(&new_list[bid][0], 1);
            unsigned id_tmp = atomicAdd(&new_num_global[bid], 1);
            // if(bid == 0) printf("%d %d %d\n", id_tmp, tid, val);
            // new_list[bid][1 + id_tmp] = val;
            hybrid_list[bid*(SAMPLE * 4)+id_tmp] = val;
            // reverse_graph[bid * RESERVENUM + 1 + id_tmp] = val;
            // if(val == 0xFFFFFFFF){
            //     printf("A\n");
            // }
        }
    }
    for(unsigned i = tid; i < old_reverse_num; i += blockDim.x * blockDim.y){
        // unsigned val = reverse_graph_old[bid][i];
        unsigned val = reverse_graph[bid * K + SAMPLE + i];
        // binary search
        unsigned tmp = old_num, res_id = 0;
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            unsigned cand = old_nn[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (old_nn[res_id] < val);
        if(old_nn[res_id] == val){
            // reverse_graph_old[bid][i] = 0xFFFFFFFF;
            reverse_graph[bid * K + SAMPLE + i] = 0xFFFFFFFF;
        }
        else{
            // unsigned id_tmp = atomicAdd(&old_list[bid][0], 1);
            unsigned id_tmp = atomicAdd(&old_num_global[bid], 1);
            // old_list[bid][1 + id_tmp] = val;
            hybrid_list[bid*(SAMPLE * 4)+2 * SAMPLE + id_tmp] = val;
        }
    }
}

__global__ void nn_descent_opt_cal(unsigned* graph, unsigned* reverse_graph, half* data, float* data_power, unsigned it, float* reverse_distance, unsigned* reverse_num, unsigned* new_num_global, unsigned* old_num_global, unsigned* hybrid_list, unsigned DIM, unsigned K){
    unsigned bid = blockIdx.x, laneid = threadIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x, warp_id_x = threadIdx.y / 4, warp_id_y = threadIdx.y % 4;
    __shared__ unsigned new_nn[SAMPLE * 2], old_nn[2* SAMPLE];
    __shared__ half new_data[SAMPLE * 2 * SS], old_data[SAMPLE * 2 * SS];

    for(unsigned i = tid; i < SAMPLE * 2 * SS; i += blockDim.x * blockDim.y){
        new_data[i] = 0.0;
        old_data[i] = 0.0;
    }

    // unsigned new_num = new_list[bid][0];
    unsigned new_num = new_num_global[bid];
    for(unsigned i = tid; i < SAMPLE * 2; i += blockDim.x * blockDim.y){
        // if(i < new_num) new_nn[i] = new_list[bid][1 + i];
        if(i < new_num) new_nn[i] = hybrid_list[bid*(SAMPLE * 4)+i];
        else new_nn[i] = 0;
    }
    // unsigned old_num = old_list[bid][0];
    unsigned old_num = old_num_global[bid];
    for(unsigned i = tid; i < SAMPLE * 2; i += blockDim.x * blockDim.y){
        // if(i < old_num) old_nn[i] = old_list[bid][1 + i];
        if(i < old_num) old_nn[i] = hybrid_list[bid*(SAMPLE * 4)+2 * SAMPLE + i];
        else old_nn[i] = 0;
    }
    // if(it == 1 && bid == 0 && tid == 0){
    //     printf("A\n");
    // }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, BLK_H, BLK_H, BLK_W, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, BLK_H, BLK_H, BLK_W, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, BLK_H, BLK_H, BLK_W, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);
    __syncthreads();
    if(old_num > 0){
    for(unsigned k = 0; k < (DIM + SS - 1) / (SS); k++){
        for(unsigned j = threadIdx.y; j < SAMPLE * 2; j += blockDim.y){
            if(j < new_num) {
                for(unsigned i = 0; i < (SS) / 32; i++){
                    new_data[j * SS + laneid + i * 32] = ((laneid + i * 32 + k * SS) < DIM ? data[new_nn[j] * DIM + laneid + i * 32 + k * SS] : __float2half(0));
                }
            }
        }
        // __syncthreads();
        for(unsigned j = threadIdx.y; j < SAMPLE * 2; j += blockDim.y){
            if(j < old_num) {
                for(unsigned i = 0; i < (SS) / 32; i++){
                    old_data[j * SS + laneid + i * 32] = ((laneid + i * 32 + k * SS) < DIM ? data[old_nn[j] * DIM + laneid + i * 32 + k * SS] : __float2half(0));
                }
            }
        }
        __syncthreads();

        for(unsigned j = 0; j < (SS) / BLK_W; j++){
            wmma::load_matrix_sync(a_frag, new_data + warp_id_x * SS * BLK_H + j * BLK_W, SS);
            wmma::load_matrix_sync(b_frag, old_data + warp_id_y * SS * BLK_H + j * BLK_W, SS);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            __syncthreads();
        }
        __syncthreads();
    }
    __syncthreads();

    for(unsigned i = 0; i < acc_frag.num_elements; i++){
        if(acc_frag.x[i] < 0.01f) acc_frag.x[i] = -INF_DIS;
    }
    acc_frag.x[0] = data_power[new_nn[warp_id_x * BLK_H + laneid/4]] + data_power[old_nn[warp_id_y * BLK_W + (laneid%4) * 2]] - 2 * acc_frag.x[0];
    acc_frag.x[1] = data_power[new_nn[warp_id_x * BLK_H + laneid/4]] + data_power[old_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 1]] - 2 * acc_frag.x[1];
    acc_frag.x[2] = data_power[new_nn[warp_id_x * BLK_H + laneid/4 + 8]] + data_power[old_nn[warp_id_y * BLK_W + (laneid%4) * 2]] - 2 * acc_frag.x[2];
    acc_frag.x[3] = data_power[new_nn[warp_id_x * BLK_H + laneid/4 + 8]] + data_power[old_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 1]] - 2 * acc_frag.x[3];
    acc_frag.x[4] = data_power[new_nn[warp_id_x * BLK_H + laneid/4]] + data_power[old_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 8]] - 2 * acc_frag.x[4];
    acc_frag.x[5] = data_power[new_nn[warp_id_x * BLK_H + laneid/4]] + data_power[old_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 9]] - 2 * acc_frag.x[5];
    acc_frag.x[6] = data_power[new_nn[warp_id_x * BLK_H + laneid/4 + 8]] + data_power[old_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 8]] - 2 * acc_frag.x[6];
    acc_frag.x[7] = data_power[new_nn[warp_id_x * BLK_H + laneid/4 + 8]] + data_power[old_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 9]] - 2 * acc_frag.x[7];
    for(unsigned i = 0; i < acc_frag.num_elements; i++){
        if(acc_frag.x[i] < 0.01f) acc_frag.x[i] = INF_DIS;
    }
    unsigned min_ele_id1, min_ele_id2;
    float min_ele_val1, min_ele_val2;

    min_ele_id1 = (acc_frag.x[0] < acc_frag.x[1] ? old_nn[warp_id_y * BLK_W + (laneid%4) * 2] : old_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 1]);
    min_ele_val1 = min(acc_frag.x[0], acc_frag.x[1]);
    min_ele_id1 = (min_ele_val1 < acc_frag.x[4] ? min_ele_id1 : old_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 8]);
    min_ele_val1 = min(min_ele_val1, acc_frag.x[4]);
    min_ele_id1 = (min_ele_val1 < acc_frag.x[5] ? min_ele_id1 : old_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 9]);
    min_ele_val1 = min(min_ele_val1, acc_frag.x[5]);
    
    min_ele_id2 = (acc_frag.x[2] < acc_frag.x[3] ? old_nn[warp_id_y * BLK_W + (laneid%4) * 2] : old_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 1]);
    min_ele_val2 = min(acc_frag.x[2], acc_frag.x[3]);
    min_ele_id2 = (min_ele_val2 < acc_frag.x[6] ? min_ele_id2 : old_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 8]);
    min_ele_val2 = min(min_ele_val2, acc_frag.x[6]);
    min_ele_id2 = (min_ele_val2 < acc_frag.x[7] ? min_ele_id2 : old_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 9]);
    min_ele_val2 = min(min_ele_val2, acc_frag.x[7]);
    
    float tmp_val1 = __shfl_down_sync(0xffffffff, min_ele_val1, 2);
    unsigned tmp_id1 = __shfl_down_sync(0xffffffff, min_ele_id1, 2);
    min_ele_id1 = (min_ele_val1 < tmp_val1 ? min_ele_id1 : tmp_id1);
    min_ele_val1 = min(min_ele_val1, tmp_val1);
    tmp_val1 = __shfl_down_sync(0xffffffff, min_ele_val1, 1);
    tmp_id1 = __shfl_down_sync(0xffffffff, min_ele_id1, 1);
    min_ele_id1 = (min_ele_val1 < tmp_val1 ? min_ele_id1 : tmp_id1);
    min_ele_val1 = min(min_ele_val1, tmp_val1);

    tmp_val1 = __shfl_down_sync(0xffffffff, min_ele_val2, 2);
    tmp_id1 = __shfl_down_sync(0xffffffff, min_ele_id2, 2);
    min_ele_id2 = (min_ele_val2 < tmp_val1 ? min_ele_id2 : tmp_id1);
    min_ele_val2 = min(min_ele_val2, tmp_val1);
    tmp_val1 = __shfl_down_sync(0xffffffff, min_ele_val2, 1);
    tmp_id1 = __shfl_down_sync(0xffffffff, min_ele_id2, 1);
    min_ele_id2 = (min_ele_val2 < tmp_val1 ? min_ele_id2 : tmp_id1);
    min_ele_val2 = min(min_ele_val2, tmp_val1);
    if(laneid % 4 == 0) 
    {   
        unsigned id_tmp;
        if(warp_id_x * BLK_H + laneid / 4 < new_num && min_ele_val1 < INF_DIS){
            id_tmp = atomicAdd(&reverse_num[new_nn[warp_id_x * BLK_H + laneid / 4]], 1);
            if(id_tmp < RESERVENUM){
                reverse_graph[new_nn[warp_id_x * BLK_H + laneid / 4] * RESERVENUM + id_tmp] = min_ele_id1;
                reverse_distance[new_nn[warp_id_x * BLK_H + laneid / 4]*RESERVENUM+id_tmp] = min_ele_val1;
            }
        }
        if(warp_id_x * BLK_H + laneid / 4 + 8 < new_num && min_ele_val2 < INF_DIS){
            id_tmp = atomicAdd(&reverse_num[new_nn[warp_id_x * BLK_H + laneid / 4 + 8]], 1);
            if(id_tmp < RESERVENUM){
                reverse_graph[new_nn[warp_id_x * BLK_H + laneid / 4 + 8] * RESERVENUM + id_tmp] = min_ele_id2;
                reverse_distance[new_nn[warp_id_x * BLK_H + laneid / 4 + 8]*RESERVENUM+id_tmp] = min_ele_val2;
            }
            // else{
            //     if(min_ele_val2 < g_tmp_res_dis[new_nn[warp_id_x * BLK_H + laneid / 4 + 8]][id_tmp % RESERVENUM]){
            //         g_tmp_res[new_nn[warp_id_x * BLK_H + laneid / 4 + 8]][id_tmp % RESERVENUM] = min_ele_id2;
            //         g_tmp_res_dis[new_nn[warp_id_x * BLK_H + laneid / 4 + 8]][id_tmp % RESERVENUM] = min_ele_val2;
            //     }
            // }
        }
    }
    __syncthreads();
    min_ele_id1 = (acc_frag.x[0] < acc_frag.x[2] ? new_nn[warp_id_x * BLK_H + laneid/4] : new_nn[warp_id_x * BLK_H + laneid/4 + 8]);
    min_ele_val1 = min(acc_frag.x[0], acc_frag.x[2]);

    min_ele_id2 = (acc_frag.x[1] < acc_frag.x[3] ? new_nn[warp_id_x * BLK_H + laneid/4] : new_nn[warp_id_x * BLK_H + laneid/4 + 8]);
    min_ele_val2 = min(acc_frag.x[1], acc_frag.x[3]);

    for(unsigned i = 0; i < 7; i++){
        tmp_val1 = __shfl_down_sync(0xffffffff, min_ele_val1, 4);
        tmp_id1 = __shfl_down_sync(0xffffffff, min_ele_id1, 4);
        min_ele_id1 = (min_ele_val1 < tmp_val1 ? min_ele_id1 : tmp_id1);
        min_ele_val1 = min(min_ele_val1, tmp_val1);
    }
    for(unsigned i = 0; i < 7; i++){
        tmp_val1 = __shfl_down_sync(0xffffffff, min_ele_val2, 4);
        tmp_id1 = __shfl_down_sync(0xffffffff, min_ele_id2, 4);
        min_ele_id2 = (min_ele_val2 < tmp_val1 ? min_ele_id2 : tmp_id1);
        min_ele_val2 = min(min_ele_val2, tmp_val1);
    }
    if(laneid / 4 == 0){
        unsigned id_tmp;
        if(warp_id_y * BLK_H + (laneid % 4) * 2 < old_num && min_ele_val1 < INF_DIS){
            id_tmp = atomicAdd(&reverse_num[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2]], 1);
            if(id_tmp < RESERVENUM){
                reverse_graph[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2] * RESERVENUM + id_tmp] = min_ele_id1;
                reverse_distance[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2]*RESERVENUM + id_tmp] = min_ele_val1;
            }
            // else{
            //     if(min_ele_val1 < g_tmp_res_dis[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2]][id_tmp % RESERVENUM]){
            //         g_tmp_res[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2]][id_tmp % RESERVENUM] = min_ele_id1;
            //         g_tmp_res_dis[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2]][id_tmp % RESERVENUM] = min_ele_val1;
            //     }
            // }
        }
        if(warp_id_y * BLK_H + (laneid % 4) * 2 + 1 < old_num && min_ele_val2 < INF_DIS){
            id_tmp = atomicAdd(&reverse_num[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 1]], 1);
            if(id_tmp < RESERVENUM){
                reverse_graph[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 1] * RESERVENUM + id_tmp] = min_ele_id2;
                reverse_distance[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 1]*RESERVENUM+id_tmp] = min_ele_val2;
            }
            // else{
            //     if(min_ele_val2 < g_tmp_res_dis[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 1]][id_tmp % RESERVENUM]){
            //         g_tmp_res[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 1]][id_tmp % RESERVENUM] = min_ele_id2;
            //         g_tmp_res_dis[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 1]][id_tmp % RESERVENUM] = min_ele_val2;
            //     }
            // }
        }
    }

    min_ele_id1 = (acc_frag.x[4] < acc_frag.x[6] ? new_nn[warp_id_x * BLK_H + laneid/4] : new_nn[warp_id_x * BLK_H + laneid/4 + 8]);
    min_ele_val1 = min(acc_frag.x[4], acc_frag.x[6]);

    min_ele_id2 = (acc_frag.x[5] < acc_frag.x[7] ? new_nn[warp_id_x * BLK_H + laneid/4] : new_nn[warp_id_x * BLK_H + laneid/4 + 8]);
    min_ele_val2 = min(acc_frag.x[5], acc_frag.x[7]);

    for(unsigned i = 0; i < 7; i++){
        tmp_val1 = __shfl_down_sync(0xffffffff, min_ele_val1, 4);
        tmp_id1 = __shfl_down_sync(0xffffffff, min_ele_id1, 4);
        min_ele_id1 = (min_ele_val1 < tmp_val1 ? min_ele_id1 : tmp_id1);
        min_ele_val1 = min(min_ele_val1, tmp_val1);
    }
    for(unsigned i = 0; i < 7; i++){
        tmp_val1 = __shfl_down_sync(0xffffffff, min_ele_val2, 4);
        tmp_id1 = __shfl_down_sync(0xffffffff, min_ele_id2, 4);
        min_ele_id2 = (min_ele_val2 < tmp_val1 ? min_ele_id2 : tmp_id1);
        min_ele_val2 = min(min_ele_val2, tmp_val1);
    }
    if(laneid / 4 == 0){
        unsigned id_tmp;
        if(warp_id_y * BLK_H + (laneid % 4) * 2 + 8 < old_num && min_ele_val1 < INF_DIS){
            id_tmp = atomicAdd(&reverse_num[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 8]], 1);
            if(id_tmp < RESERVENUM){
                reverse_graph[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 8] * RESERVENUM + id_tmp] = min_ele_id1;
                reverse_distance[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 8]*RESERVENUM+id_tmp] = min_ele_val1;
            }
            // else{
            //     if(min_ele_val1 < g_tmp_res_dis[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 8]][id_tmp % RESERVENUM]){
            //         g_tmp_res[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 8]][id_tmp % RESERVENUM] = min_ele_id1;
            //         g_tmp_res_dis[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 8]][id_tmp % RESERVENUM] = min_ele_val1;
            //     }
            // }
        }

        if(warp_id_y * BLK_H + (laneid % 4) * 2 + 9 < old_num && min_ele_val2 < INF_DIS){
            id_tmp = atomicAdd(&reverse_num[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 9]], 1);
            if(id_tmp < RESERVENUM){
                reverse_graph[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 9] * RESERVENUM + id_tmp] = min_ele_id2;
                reverse_distance[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 9]*RESERVENUM+id_tmp] = min_ele_val2;
            }
            // else{
            //     if(min_ele_val2 < g_tmp_res_dis[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 9]][id_tmp % RESERVENUM]){
            //         g_tmp_res[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 9]][id_tmp % RESERVENUM] = min_ele_id2;
            //         g_tmp_res_dis[old_nn[warp_id_y * BLK_H + (laneid % 4) * 2 + 9]][id_tmp % RESERVENUM] = min_ele_val2;
            //     }
            // }
        }
    }
    }
    __syncthreads();
    for(unsigned i = tid; i < SAMPLE * 2 * SS; i += blockDim.x * blockDim.y){
        new_data[i] = 0.0;
    }
    __syncthreads();

    wmma::fill_fragment(acc_frag, 0.0);
    for(unsigned k = 0; k < (DIM + SS - 1) / (SS); k++){
        for(unsigned j = threadIdx.y; j < SAMPLE * 2; j += blockDim.y){
            if(j < new_num) {
                for(unsigned i = 0; i < (SS) / 32; i++){
                    new_data[j * SS + laneid + i * 32] = ((laneid + i * 32 + k * SS) < DIM ? data[new_nn[j] * DIM + laneid + i * 32 + k * SS] : __float2half(0));
                    // if(j * DIM + laneid + i * 32 > SAMPLE * 2 * SAMPLE * 2) printf("A");
                    // if(new_nn[j] * DIM + laneid + i * 32 + k * SAMPLE * 2 > 1000000 * DIM) printf("B");
                }
            }
        }
        __syncthreads();
        // if(bid == 0 && tid == 0) printf("%f %f\n", __half2float(new_data[0]), __half2float(new_data[1]));
        for(unsigned j = 0; j < (SS) / BLK_W; j++){
            wmma::load_matrix_sync(a_frag, new_data + warp_id_x * SS * BLK_H + j * BLK_W, SS);
            wmma::load_matrix_sync(b_frag, new_data + warp_id_y * SS * BLK_H + j * BLK_W, SS);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            __syncthreads();
        }
        __syncthreads();
    }
    __syncthreads();
    if(warp_id_x <= warp_id_y){
        // if(bid == 0 && warp_id_x == 0) printf("AA %f %f %f %f\n", acc_frag.x[0], acc_frag.x[1], acc_frag.x[2], acc_frag.x[3]);
        for(unsigned i = 0; i < acc_frag.num_elements; i++){
            if(acc_frag.x[i] < 0.01f) {
                acc_frag.x[i] = -INF_DIS;
            }
        } 
        acc_frag.x[0] = data_power[new_nn[warp_id_x * BLK_H + laneid/4]] + data_power[new_nn[warp_id_y * BLK_W + (laneid%4) * 2]] - 2 * acc_frag.x[0];
        acc_frag.x[1] = data_power[new_nn[warp_id_x * BLK_H + laneid/4]] + data_power[new_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 1]] - 2 * acc_frag.x[1];
        acc_frag.x[2] = data_power[new_nn[warp_id_x * BLK_H + laneid/4 + 8]] + data_power[new_nn[warp_id_y * BLK_W + (laneid%4) * 2]] - 2 * acc_frag.x[2];
        acc_frag.x[3] = data_power[new_nn[warp_id_x * BLK_H + laneid/4 + 8]] + data_power[new_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 1]] - 2 * acc_frag.x[3];
        acc_frag.x[4] = data_power[new_nn[warp_id_x * BLK_H + laneid/4]] + data_power[new_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 8]] - 2 * acc_frag.x[4];
        acc_frag.x[5] = data_power[new_nn[warp_id_x * BLK_H + laneid/4]] + data_power[new_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 9]] - 2 * acc_frag.x[5];
        acc_frag.x[6] = data_power[new_nn[warp_id_x * BLK_H + laneid/4 + 8]] + data_power[new_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 8]] - 2 * acc_frag.x[6];
        acc_frag.x[7] = data_power[new_nn[warp_id_x * BLK_H + laneid/4 + 8]] + data_power[new_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 9]] - 2 * acc_frag.x[7];

        for(unsigned i = 0; i < acc_frag.num_elements; i++){
            if(acc_frag.x[i] < 0.01f) acc_frag.x[i] = INF_DIS;
        }
        unsigned min_ele_id1, min_ele_id2;
        float min_ele_val1, min_ele_val2;
        
        min_ele_id1 = (acc_frag.x[0] < acc_frag.x[1] ? new_nn[warp_id_y * BLK_W + (laneid%4) * 2] : new_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 1]);
        min_ele_val1 = min(acc_frag.x[0], acc_frag.x[1]);
        min_ele_id1 = (min_ele_val1 < acc_frag.x[4] ? min_ele_id1 : new_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 8]);
        min_ele_val1 = min(min_ele_val1, acc_frag.x[4]);
        min_ele_id1 = (min_ele_val1 < acc_frag.x[5] ? min_ele_id1 : new_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 9]);
        min_ele_val1 = min(min_ele_val1, acc_frag.x[5]);
        min_ele_id2 = (acc_frag.x[2] < acc_frag.x[3] ? new_nn[warp_id_y * BLK_W + (laneid%4) * 2] : new_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 1]);
        min_ele_val2 = min(acc_frag.x[2], acc_frag.x[3]);
        min_ele_id2 = (min_ele_val2 < acc_frag.x[6] ? min_ele_id2 : new_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 8]);
        min_ele_val2 = min(min_ele_val2, acc_frag.x[6]);
        min_ele_id2 = (min_ele_val2 < acc_frag.x[7] ? min_ele_id2 : new_nn[warp_id_y * BLK_W + (laneid%4) * 2 + 9]);
        min_ele_val2 = min(min_ele_val2, acc_frag.x[7]);

        float tmp_val1 = __shfl_down_sync(0xffffffff, min_ele_val1, 2);
        unsigned tmp_id1 = __shfl_down_sync(0xffffffff, min_ele_id1, 2);
        min_ele_id1 = (min_ele_val1 < tmp_val1 ? min_ele_id1 : tmp_id1);
        min_ele_val1 = min(min_ele_val1, tmp_val1);
        tmp_val1 = __shfl_down_sync(0xffffffff, min_ele_val1, 1);
        tmp_id1 = __shfl_down_sync(0xffffffff, min_ele_id1, 1);
        min_ele_id1 = (min_ele_val1 < tmp_val1 ? min_ele_id1 : tmp_id1);
        min_ele_val1 = min(min_ele_val1, tmp_val1);

        tmp_val1 = __shfl_down_sync(0xffffffff, min_ele_val2, 2);
        tmp_id1 = __shfl_down_sync(0xffffffff, min_ele_id2, 2);
        min_ele_id2 = (min_ele_val2 < tmp_val1 ? min_ele_id2 : tmp_id1);
        min_ele_val2 = min(min_ele_val2, tmp_val1);
        tmp_val1 = __shfl_down_sync(0xffffffff, min_ele_val2, 1);
        tmp_id1 = __shfl_down_sync(0xffffffff, min_ele_id2, 1);
        min_ele_id2 = (min_ele_val2 < tmp_val1 ? min_ele_id2 : tmp_id1);
        min_ele_val2 = min(min_ele_val2, tmp_val1);

        if(laneid % 4 == 0) 
        {   
            unsigned id_tmp;
            if(warp_id_x * BLK_H + laneid / 4 < new_num){
                id_tmp = atomicAdd(&reverse_num[new_nn[warp_id_x * BLK_H + laneid / 4]], 1);
                if(id_tmp < RESERVENUM){
                    reverse_graph[new_nn[warp_id_x * BLK_H + laneid / 4] * RESERVENUM + id_tmp] = min_ele_id1;
                    reverse_distance[new_nn[warp_id_x * BLK_H + laneid / 4]*RESERVENUM+id_tmp] = min_ele_val1;
                }
                // else{
                //     if(min_ele_val1 < g_tmp_res_dis[new_nn[warp_id_x * BLK_H + laneid / 4]][id_tmp % RESERVENUM]){
                //         g_tmp_res[new_nn[warp_id_x * BLK_H + laneid / 4]][id_tmp % RESERVENUM] = min_ele_id1;
                //         g_tmp_res_dis[new_nn[warp_id_x * BLK_H + laneid / 4]][id_tmp % RESERVENUM] = min_ele_val1;
                //     }
                // }
            }
            
            if(warp_id_x * BLK_H + laneid / 4 + 8 < new_num){
                id_tmp = atomicAdd(&reverse_num[new_nn[warp_id_x * BLK_H + laneid / 4 + 8]], 1);
                if(id_tmp < RESERVENUM){
                    reverse_graph[new_nn[warp_id_x * BLK_H + laneid / 4 + 8] * RESERVENUM + id_tmp] = min_ele_id2;
                    reverse_distance[new_nn[warp_id_x * BLK_H + laneid / 4 + 8]*RESERVENUM+id_tmp] = min_ele_val2;
                }
                // else{
                //     if(min_ele_val2 < g_tmp_res_dis[new_nn[warp_id_x * BLK_H + laneid / 4 + 8]][id_tmp % RESERVENUM]){
                //         g_tmp_res[new_nn[warp_id_x * BLK_H + laneid / 4 + 8]][id_tmp % RESERVENUM] = min_ele_id2;
                //         g_tmp_res_dis[new_nn[warp_id_x * BLK_H + laneid / 4 + 8]][id_tmp % RESERVENUM] = min_ele_val2;
                //     }
                // }
            }

        }
    }
}

__global__ void nn_descent_opt_merge(unsigned* graph, unsigned* reverse_graph, unsigned it, unsigned all_it, float* nei_distance, float* reverse_distance, bool* nei_visit, unsigned* reverse_num, unsigned K){
    unsigned bid = blockIdx.x, laneid = threadIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ unsigned new_list_shared[RESERVENUM], new_list2[RESERVENUM], nei_list_shared[K_SIZE];
    __shared__ float new_dist_shared[RESERVENUM], nei_dist_shared[K_SIZE];
    __shared__ bool nei_visit_shared[K_SIZE];
    float min_ele = nei_distance[bid*K+K - 1];
    __shared__ unsigned dup_num;
    if(tid == 0) dup_num = 0;
    __syncthreads();
    unsigned new_num = min(RESERVENUM, reverse_num[bid]);
    for(unsigned i = tid; i < new_num; i += blockDim.x * blockDim.y){
        if(reverse_distance[bid*RESERVENUM+i] < min_ele){
            unsigned id_tmp = atomicAdd(&dup_num, 1);
            new_list_shared[id_tmp] = reverse_graph[bid * RESERVENUM + i];
            new_dist_shared[id_tmp] = reverse_distance[bid*RESERVENUM+i];
        }
    }

    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        nei_visit_shared[i] = nei_visit[bid*K+i];
        nei_dist_shared[i] = nei_distance[bid*K+i];
        nei_list_shared[i] = graph[bid * K + i];
    }

    for(unsigned i = tid; i < RESERVENUM; i+= blockDim.x * blockDim.y){
        new_list2[i] = 0;
    }

    __syncthreads();
    bitonic_sort_by_id(new_dist_shared, new_list_shared, dup_num);
    for(unsigned i = tid + 1; i < dup_num; i += blockDim.x * blockDim.y){
        if(new_list_shared[i] == new_list_shared[i - 1] || new_list_shared[i] == bid){
            new_list2[i] = K;
            new_dist_shared[i] = INF_DIS;
        }
    }
    if(tid == 0 && new_list_shared[0] == bid){
        new_list2[0] = K;
        new_dist_shared[0] = INF_DIS;
    }
    __syncthreads();
   
    if(bid == 0 && tid == 0) printf("Dup: %d %d\n", dup_num, it);

    for(unsigned i = tid; i < dup_num; i += blockDim.x * blockDim.y){
        if(new_list2[i] == K){
            continue;
        }
        float val = new_dist_shared[i];
        // binary search
        unsigned tmp = K, res_id = 0;
        while (tmp > 1) {
            unsigned halfsize = tmp / 2;
            float cand = nei_dist_shared[res_id + halfsize];
            res_id += ((cand < val) ? halfsize : 0);
            tmp -= halfsize;
        }
        res_id += (nei_dist_shared[res_id] < val);
        unsigned tmp_count = 0;
        while(res_id + tmp_count < K && nei_dist_shared[res_id + tmp_count] == new_dist_shared[i]){
            if(nei_list_shared[res_id + tmp_count] == new_list_shared[i]){
                new_dist_shared[i] = INF_DIS;
                res_id = K;
                break;
            }
            tmp_count++;
        }
        new_list2[i] = res_id;
    }
    __syncthreads();

    
    bitonic_sort_new2(new_dist_shared, new_list_shared, new_list2, dup_num);

    // merge
    if(dup_num > 0){
        for(unsigned i = threadIdx.y; i < (K + dup_num + blockDim.x - 1) / blockDim.x; i += blockDim.y){
            unsigned res_id = 0, id_reg;
            bool visit_reg = false;
            float val;
            if(i < K / blockDim.x){
                val = nei_dist_shared[laneid + i * blockDim.x];
                id_reg = nei_list_shared[laneid + i * blockDim.x];
                visit_reg = nei_visit_shared[laneid + i * blockDim.x];
                unsigned tmp = dup_num;
                while (tmp > 1) {
                    unsigned halfsize = tmp / 2;
                    float cand = new_dist_shared[res_id + halfsize];
                    res_id += ((cand <= val) ? halfsize : 0);
                    tmp -= halfsize;
                }
                res_id += (new_dist_shared[res_id] <= val);
                res_id += (laneid + i * blockDim.x);
            }
            else{
                if(laneid + i * blockDim.x - K < dup_num){
                    val = new_dist_shared[laneid + i * blockDim.x - K];
                    id_reg = new_list_shared[laneid + i * blockDim.x - K];
                    res_id = (new_list2[laneid + i * blockDim.x - K] + laneid + i * blockDim.x - K);
                }
                else{
                    res_id = K;
                }
            }
            __syncthreads();
            if(res_id < K){
                nei_distance[bid*K+res_id] = val;
                graph[bid * K + res_id] = id_reg;
                nei_visit[bid*K+res_id] = visit_reg;
            }
        }
    }
    __syncthreads();
    if(it == all_it - 1 && bid == 0 && tid == 0){
        for(unsigned i = 0; i < K; i++)
            printf("%f, ", nei_distance[bid * K + i]);
        printf("\n");
    }
}

__global__ void cal_power(half* data_half, float* data_power, unsigned dim, unsigned DIM){
    unsigned bid = blockIdx.x, tid = threadIdx.x;
    __shared__ half shared_val[(DIM_SIZE + 31) / 32];

    half val_res = data_half[bid * dim + tid];
    val_res = __hmul(val_res, val_res);
    #pragma unroll
    for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
        val_res = __hadd(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
    }
    if(tid % 32 == 0) shared_val[tid / 32] = val_res;
    __syncthreads();
    if(tid == 0){
        val_res = 0.0;
        for(unsigned i = 0; i < (DIM + 31) / 32; i++){
            val_res = __hadd(val_res, shared_val[i]);
        }
        data_power[bid] = __half2float(val_res);
    }

}

__global__ void do_reverse_graph(unsigned* graph_dev, unsigned* reverse_graph_dev, float* nei_distance, float* reverse_distance, unsigned* reverse_num, unsigned K){
    unsigned bid = blockIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x;
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        unsigned to_add = graph_dev[bid * K + i];
        unsigned tmp_id = atomicAdd(&reverse_num[to_add], 1);
        if(tmp_id < K) {
            reverse_graph_dev[to_add * K + tmp_id] = bid;
            reverse_distance[to_add*RESERVENUM+tmp_id] = nei_distance[bid*K+i];
        }
    }
}

__global__ void reset_visit_reversenum(unsigned* reverse_graph, bool* nei_visit, unsigned* reverse_num, unsigned POINTS, unsigned K){
    unsigned bid = blockIdx.x, tid = threadIdx.x + threadIdx.y * blockDim.x;
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        nei_visit[bid*K+i] = false;
        if(reverse_graph[bid * K + i] > POINTS) reverse_graph[bid * K + i] = 0;
    }
    if(tid == 0) reverse_num[bid] = 0;
}

__device__ void collect_path(unsigned* shared_cand, float* shared_dis, unsigned* final_nei, float* final_dis, unsigned* top_M_Cand, unsigned* top_M_Cand2, float* top_M_Cand_dis, bool* has_explore, half4* tmp_val_sha, unsigned tid, unsigned bid, unsigned laneid, unsigned K, unsigned DIM, unsigned TOPM, unsigned* ep, unsigned* graph, const half* __restrict__  values, unsigned max_cand, float* nei_distance){
    __shared__ unsigned curr_of_candset, to_explore;
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        shared_cand[K + tid] = graph[(*ep) * K + tid];
        top_M_Cand[TOPM + tid] = graph[(*ep) * K + tid];
        has_explore[TOPM + tid] = false;
    }
    if(tid == 0) {
        curr_of_candset = K;
    }
    __syncthreads();

    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&values[bid * DIM + 4 * i]);
	}
	__syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < K; i += blockDim.y){
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
        }

        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
        }

        if(laneid == 0){
            top_M_Cand_dis[TOPM + i] = __half2float(__hadd(val_res.x, val_res.y));
            shared_dis[K + i] = top_M_Cand_dis[TOPM + i];
        }
    }
    __syncthreads();

    bitonic_sort_id_by_dis(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], &has_explore[TOPM], K);
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = top_M_Cand[TOPM + i];
        top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
        has_explore[i] = has_explore[TOPM + i];
    }
    for(unsigned i = tid; i < TOPM + K; i += blockDim.x * blockDim.y){
        top_M_Cand2[i] = top_M_Cand[i];
    }

    __syncthreads();

    bitonic_sort_id_new2(top_M_Cand2, TOPM + K);
    // begin search
    for(unsigned i = 0; i < 10; i++){
        // explore the first node
        if(tid == 0){
            for(unsigned j = 0; j < TOPM; j++){
                if(has_explore[j] == false){
                    to_explore = top_M_Cand[j];
                    has_explore[j] = true;
                    break;
                }
                if(j == TOPM) {
                    to_explore = 0xFFFFFFFF;
                }
            }
        }
        __syncthreads();
        if(to_explore == 0xFFFFFFFF) {
            break;
        }
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            unsigned to_append = graph[to_explore * K + j];
            // remove duplicate
            unsigned tmp = TOPM + K, res_id = 0;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                unsigned cand = top_M_Cand2[res_id + halfsize];
                res_id += ((cand <= to_append) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id += (top_M_Cand2[res_id] < to_append);
            if(res_id < K + TOPM && top_M_Cand2[res_id] == to_append){
                top_M_Cand_dis[TOPM + j] = INF_DIS;
            }
            else{
                top_M_Cand_dis[TOPM + j] = 0.0;
            }
            top_M_Cand[TOPM + j] = to_append;
            has_explore[TOPM + j] = false;
        }
        __syncthreads();
        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < K; j += blockDim.y){
            if(top_M_Cand_dis[TOPM + j] < 1.0){
                
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                }
                
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    top_M_Cand_dis[TOPM + j] = __half2float(__hadd(val_res.x, val_res.y));
                    unsigned tmp_res = atomicAdd(&curr_of_candset, 1);
                    shared_cand[K + tmp_res % max_cand] = top_M_Cand[TOPM + j];
                    shared_dis[K + tmp_res % max_cand] = top_M_Cand_dis[TOPM + j];
                }
            }
        }
        __syncthreads();
        // merge into top_M
        bitonic_sort_id_by_dis(top_M_Cand_dis, top_M_Cand, has_explore, TOPM + K);
        for(unsigned i = tid; i < TOPM + K; i += blockDim.x * blockDim.y){
            top_M_Cand2[i] = top_M_Cand[i];
        }
        __syncthreads();

        bitonic_sort_id_new2(top_M_Cand2, TOPM + K);
    }
    for(unsigned i = curr_of_candset + tid; i < max_cand; i += blockDim.x * blockDim.y){
        shared_dis[K + i] = INF_DIS;
        shared_cand[K + i] = 0xFFFFFFFF;
    }
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        shared_cand[i] = graph[bid * K + i];
        shared_dis[i] = nei_distance[bid*K+i];
    }
    __syncthreads();
    bitonic_sort_id_and_dis(shared_dis, shared_cand, SELECT_CAND+K);
    for(unsigned i = tid + 1; i < SELECT_CAND+K; i += blockDim.x * blockDim.y){
        if(shared_cand[i] == shared_cand[i - 1]) shared_cand[i] = 0xFFFFFFFF;
    }
}

__device__ void collect_2hop(unsigned* shared_cand, float* shared_dis, half4* tmp_val_sha, unsigned tid, unsigned bid, unsigned laneid, unsigned K, unsigned DIM, unsigned* graph, const half* __restrict__  values, float* nei_distance){
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        shared_cand[i] = graph[bid * K + i];
        shared_dis[i] = nei_distance[bid*K + i];
    }
    for(unsigned i = tid; i < SELECT_CAND; i += blockDim.x * blockDim.y){
        shared_cand[K + i] = graph[graph[bid * K + i / K] * K + i % K];
    }
    __syncthreads();
    // half4 val1 = Load(&values[bid * DIM + 4 * laneid]);
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&values[bid * DIM + 4 * i]);
	}
    __syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < SELECT_CAND; i += blockDim.y){
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;

        // half4 val2 = Load(&values[shared_cand[K + i] * DIM + laneid * 4]);
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            half4 val2 = Load(&values[shared_cand[K + i] * DIM + j * 4]);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
        }
        
        
        // val_res = __hmul2(__hsub2(val1.x, val2.x), __hsub2(val1.x, val2.x));
        // val_res = __hfma2(__hsub2(val1.y, val2.y), __hsub2(val1.y, val2.y), val_res);
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
        }

        if(laneid == 0){
            shared_dis[K + i] = __half2float(__hadd(val_res.x, val_res.y));
        }
    }
    __syncthreads();
    bitonic_sort_id_and_dis(shared_dis, shared_cand, SELECT_CAND+K);

    for(unsigned i = tid + 1; i < SELECT_CAND+K; i += blockDim.x * blockDim.y){
        if(shared_cand[i] == shared_cand[i - 1]) shared_cand[i] = 0xFFFFFFFF;
    }
}

__device__ void Filter_path(unsigned* shared_cand, float* shared_dis, unsigned* final_nei, float* final_dis, half4* tmp_val_sha, unsigned tid, unsigned bid, unsigned laneid, unsigned K, unsigned DIM, unsigned FINAL_DEGREE, unsigned* graph, const half* __restrict__  values, float* nei_distance, unsigned* reverse_num, unsigned* reverse_graph, float* reverse_distance, float (*operat)(float,float,float,float), float threshold){
    __shared__ unsigned cur_nei;
    __shared__ unsigned tmp_id;
    __shared__ float cur_nei_dis;
    if(tid == 0) {
        tmp_id = 0;
        while(shared_cand[tmp_id] == bid || shared_cand[tmp_id] == 0xFFFFFFFF) tmp_id++;
        final_nei[0] = shared_cand[tmp_id];
        final_dis[0] = shared_dis[tmp_id];
        cur_nei_dis = shared_dis[tmp_id];
        cur_nei = 1;
        tmp_id++;
    }
    
    __syncthreads();
    while(cur_nei < FINAL_DEGREE){
        for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
            tmp_val_sha[i] = Load(&values[final_nei[cur_nei - 1] * DIM + 4 * i]);
        }
        __syncthreads();
        unsigned i;
        for(i = tmp_id + threadIdx.y; i < SELECT_CAND + K; i += blockDim.y){
            if(shared_cand[i] == 0xFFFFFFFF || shared_cand[i] == final_nei[cur_nei - 1]) 
            {
                shared_cand[i] = 0xFFFFFFFF;
                continue;
            }

            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                half4 val2 = Load(&values[shared_cand[i] * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }

            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }

            if(laneid == 0){
                float res = __half2float(__hadd(val_res.x, val_res.y));
                if(operat(res, shared_dis[i], cur_nei_dis, threshold)<0.0f){
                    shared_cand[i] = 0xFFFFFFFF;
                }
            }
        }
        __syncthreads();
        if(tid == 0){
            while(tmp_id < SELECT_CAND + K && shared_cand[tmp_id] == 0xFFFFFFFF) tmp_id++;
            final_nei[cur_nei] = shared_cand[tmp_id];
            final_dis[cur_nei] = shared_dis[tmp_id];
            cur_nei_dis = shared_dis[tmp_id];
            cur_nei++;
            tmp_id++;
        }
        __syncthreads();
        if(tmp_id >= SELECT_CAND+K) break;
    }
    __syncthreads();
    for(unsigned i = tid; i < cur_nei - 1; i += blockDim.x * blockDim.y){
        graph[bid * K + i + 1] = final_nei[i];
        nei_distance[bid*K+i] = final_dis[i];
        unsigned tmp_rev = atomicAdd(&reverse_num[final_nei[i]], 1);
        if(tmp_rev < K){
            reverse_graph[final_nei[i] * K + tmp_rev] = bid;
            reverse_distance[final_nei[i]*RESERVENUM+tmp_rev] = final_dis[i];
        }
        else{
            if(final_dis[i] < reverse_distance[final_nei[i]*RESERVENUM+(tmp_rev % K)]){
                reverse_graph[final_nei[i] * K + tmp_rev % K] = bid;
                reverse_distance[final_nei[i]*RESERVENUM+(tmp_rev % K)] = final_dis[i];
            }
        }
    }
    if(tid == 0) graph[bid * K] = cur_nei - 1;
    if(bid == 0 && tid == 0) printf("Cur_nei: %d; tmpid: %d\n", cur_nei, tmp_id);
    if(bid == 0 && tid == 0) {
        for(unsigned i = 0; i < cur_nei-1; i++){
            printf("%d,", final_nei[i]);
        }
        printf("\n");
    }
}

__device__ float distance_filter(float res, float dis, float cur_dis, float threshold){
    return ((threshold * res) - dis);
}

__device__ float angle_filter(float res, float dis, float cur_dis, float threshold){
    return threshold - (dis + cur_dis - res) / 2 / sqrtf(dis * cur_dis);
}


__global__ void select_path(unsigned* graph, unsigned* reverse_graph, unsigned *ep, const half* __restrict__ values, unsigned max_cand, float* nei_distance, float* reverse_distance, unsigned* reverse_num, unsigned DIM, unsigned FINAL_DEGREE, unsigned TOPM, unsigned K, float thre){
    __shared__ unsigned shared_cand[SELECT_CAND+K_SIZE];
    __shared__ float shared_dis[SELECT_CAND+K_SIZE];
    __shared__ unsigned final_nei[FINAL_DEGREE_SIZE];
    __shared__ float final_dis[FINAL_DEGREE_SIZE];
    
    __shared__ unsigned top_M_Cand[TOPM_SIZE+K_SIZE];
    __shared__ unsigned top_M_Cand2[TOPM_SIZE+K_SIZE];
    __shared__ float top_M_Cand_dis[TOPM_SIZE+K_SIZE];
    __shared__ bool has_explore[TOPM_SIZE+K_SIZE];
    __shared__ half4 tmp_val_sha[DIM_SIZE / 4];
    
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;
    // initialize
    collect_path(shared_cand, shared_dis, final_nei, final_dis, top_M_Cand, top_M_Cand2, top_M_Cand_dis, has_explore, tmp_val_sha, tid, bid, laneid, K, DIM, TOPM, ep, graph, values, max_cand, nei_distance);
    __syncthreads();

    Filter_path(shared_cand, shared_dis, final_nei, final_dis, tmp_val_sha, tid, bid, laneid, K, DIM, FINAL_DEGREE, graph, values, nei_distance, reverse_num, reverse_graph, reverse_distance, distance_filter, thre);
    
}

__global__ void select_2hop(unsigned* graph, unsigned* reverse_graph, const half* __restrict__ values, unsigned max_cand, float* nei_distance, float* reverse_distance, unsigned* reverse_num, unsigned DIM, unsigned FINAL_DEGREE, unsigned TOPM, unsigned K, float thre){
    __shared__ unsigned shared_cand[SELECT_CAND+K_SIZE];
    __shared__ float shared_dis[SELECT_CAND+K_SIZE];
    __shared__ unsigned final_nei[FINAL_DEGREE_SIZE];
    __shared__ float final_dis[FINAL_DEGREE_SIZE];
    
    __shared__ half4 tmp_val_sha[DIM_SIZE / 4];
    
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;
    // initialize
    collect_2hop(shared_cand, shared_dis, tmp_val_sha, tid, bid, laneid, K, DIM, graph, values, nei_distance);
    __syncthreads();

    Filter_path(shared_cand, shared_dis, final_nei, final_dis, tmp_val_sha, tid, bid, laneid, K, DIM, FINAL_DEGREE, graph, values, nei_distance, reverse_num, reverse_graph, reverse_distance, angle_filter, thre);
}

__global__ void filter_reverse(unsigned* graph, unsigned* reverse_graph, half* values, float* nei_distance, float* reverse_distance, unsigned* reverse_num, unsigned DIM, unsigned FINAL_DEGREE, unsigned K, funcFormat operat, float threshold){
    __shared__ unsigned shared_cand[SELECT_CAND+K_SIZE];
    __shared__ float shared_dis[SELECT_CAND+K_SIZE];
    __shared__ unsigned final_nei[FINAL_DEGREE_SIZE];
    __shared__ unsigned final_dis[FINAL_DEGREE_SIZE];
    __shared__ half4 tmp_val_sha[DIM_SIZE / 4];
    __shared__ unsigned cur_nei;
    __shared__ unsigned tmp_id;
    __shared__ float cur_nei_dis;
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;
    
    unsigned cur_num = graph[bid * K], rever_num = min(K, reverse_num[bid]);

    for(unsigned i = tid; i < cur_num; i += blockDim.x * blockDim.y){
        shared_cand[i] = graph[bid * K + i + 1];
        shared_dis[i] = nei_distance[bid*K+i];
     }
    __syncthreads();
    for(unsigned i = tid; i < rever_num; i += blockDim.x * blockDim.y){
        shared_cand[cur_num + i] = reverse_graph[bid * K + i];
        shared_dis[cur_num + i] = reverse_distance[bid*RESERVENUM+i];
    }
    __syncthreads();

    if(cur_num + rever_num > FINAL_DEGREE){
        bitonic_sort_id_and_dis(shared_dis, shared_cand, cur_num+rever_num);
        for(unsigned i = tid + 1; i < cur_num+rever_num; i += blockDim.x * blockDim.y){
            if(shared_cand[i] == shared_cand[i - 1]) shared_cand[i] = 0xFFFFFFFF;
        }
        __syncthreads();
        if(tid == 0) {
            tmp_id = 0;
            while(shared_cand[tmp_id] == bid || shared_cand[tmp_id] == 0xFFFFFFFF) tmp_id++;
            final_nei[0] = shared_cand[tmp_id];
            final_dis[0] = shared_dis[tmp_id];
            cur_nei = 1;
            tmp_id++;
        }
        
        __syncthreads();
        while(cur_nei < FINAL_DEGREE){
            for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
                tmp_val_sha[i] = Load(&values[final_nei[cur_nei - 1] * DIM + 4 * i]);
            }
            __syncthreads();
            unsigned i;
            for(i = tmp_id + threadIdx.y; i < cur_num + rever_num; i += blockDim.y){
                if(shared_cand[i] == 0xFFFFFFFF || shared_cand[i] == final_nei[cur_nei - 1]) 
                {
                    shared_cand[i] = 0xFFFFFFFF;
                    continue;
                }

                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;

                for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                    half4 val2 = Load(&values[shared_cand[i] * DIM + j * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
                }

                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    float res = __half2float(__hadd(val_res.x, val_res.y));
                    if(operat(res, shared_dis[i], cur_nei_dis, threshold)<0.0f){
                        shared_cand[i] = 0xFFFFFFFF;
                    }
                }
            }
            __syncthreads();
            if(tid == 0){
                while(tmp_id < cur_num + rever_num && shared_cand[tmp_id] == 0xFFFFFFFF) tmp_id++;
                final_nei[cur_nei] = shared_cand[tmp_id];
                final_dis[cur_nei] = shared_dis[tmp_id];
                cur_nei++;
                tmp_id++;
            }
            __syncthreads();
            if(tmp_id >= cur_num+rever_num) break;
        }
        __syncthreads();
        for(unsigned i = tid; i < cur_nei - 1; i += blockDim.x * blockDim.y){
            graph[bid * K + i + 1] = final_nei[i];
        }
        if(tid == 0) graph[bid * K] = cur_nei - 1;
    }
    else{
        __shared__ unsigned cur_dup_num;
        if(tid == 0) cur_dup_num = 0;
        __syncthreads();
        for(unsigned i = tid; i < cur_num; i += blockDim.x * blockDim.y){
            graph[bid * K + i + 1] = shared_cand[i];
        }
        __syncthreads();
        for(unsigned i = tid; i < rever_num; i += blockDim.x * blockDim.y){
            unsigned j = 0;
            for(j = 0; j < cur_num; j++){
                if(shared_cand[j] == shared_cand[cur_num + i]) break;
            }
            if(j >= cur_num){
                unsigned t_id = atomicAdd(&cur_dup_num, 1);
                graph[bid * K + cur_num + t_id + 1] = shared_cand[cur_num + i];
            }
        }
        __syncthreads();
        if(tid == 0) graph[bid * K] = cur_num + cur_dup_num;
        __syncthreads();
    }

    if(bid == 0 && tid == 0) {
        for(unsigned i = 0; i < graph[bid * K]; i++){
            printf("%d,", graph[bid * K + i + 1]);
        }
        printf("\n");
    }
}

__device__ void collect_1hop(unsigned * graph, unsigned K, unsigned* nei0, unsigned* detour_num, unsigned tid, unsigned bid){
    for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
        nei0[j] = graph[bid * K + j];
        detour_num[j] = 0;
    }
}

__device__ void filter_cagra(unsigned* nei0, unsigned* nei1, unsigned* nei2, unsigned* detour_num, unsigned* graph, unsigned K, unsigned d, unsigned* reverse_num, unsigned* reverse_graph, unsigned* new_list, unsigned tid, unsigned bid){
    for(unsigned i = 0; i < K / 2; i++){
        unsigned nei_id1 = graph[bid * K + i], nei_id2 = graph[bid * K + K - 1 - i];
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            nei1[j] = graph[nei_id1 * K + j];
            nei2[j] = graph[nei_id2 * K + j];
        }
        __syncthreads();
        unsigned tmp = K, res_id = 0;
        if(tid < K - 1){
            unsigned find_key = ((tid < K - 1 - i) ? nei0[i + 1 + tid] : nei0[tid + 1]);
            unsigned* shared_nei = ((tid < K - 1 - i) ? &nei1[0] : &nei2[0]);
            for(unsigned j = 0; j < K; j++){
                if(shared_nei[j] == find_key) {
                    res_id = j;
                    break;
                }
            }
            if(res_id < K && shared_nei[res_id] == find_key){
                unsigned ori_dis = ((tid < K - 1 - i) ? i : (K - 1 - i)), my_dis = ((tid < K - 1 - i) ? (i + 1 + tid) : (tid + 1));
                // if(max(ori_dis, res_id) < my_dis){
                    atomicAdd(&detour_num[my_dis], 1);
                // }
            }
        }
        __syncthreads();
    }
    bitonic_sort_id_by_detour(detour_num, nei0, K);
    for(unsigned i = tid; i < d; i += blockDim.x * blockDim.y){
        unsigned tmp = atomicAdd(&reverse_num[nei0[i]], 1);
        if (tmp < d) {
            reverse_graph[nei0[i] * d * 2 + tmp * 2] = bid;
            reverse_graph[nei0[i] * d * 2 + tmp * 2 + 1] = detour_num[i];
        }
        else{
            if(reverse_graph[nei0[i] * d * 2 + (tmp%d) * 2 + 1] > detour_num[i]){
                reverse_graph[nei0[i] * d * 2 + (tmp%d) * 2] = bid;
                reverse_graph[nei0[i] * d * 2 + (tmp%d) * 2 + 1] = detour_num[i];
            }
        }
        new_list[bid*K+i] = nei0[i];
    }
}

__device__ void filter_dpg(unsigned* nei0, unsigned* nei1, unsigned* nei2, unsigned* detour_num, unsigned* graph, unsigned K, unsigned d, unsigned* reverse_num, unsigned* reverse_graph, unsigned* new_list, unsigned tid, unsigned bid, unsigned laneid, unsigned DIM, const half* __restrict__ values, float* nei_distance){
    __shared__ half4 tmp_val_sha[DIM_SIZE / 4];
    for(unsigned i = threadIdx.y; i < K / 2; i += blockDim.y){
        unsigned nei_id1 = nei0[i], nei_id2 = nei0[K - 1 - i];
        for(unsigned i = laneid; i < (DIM / 4); i += blockDim.x){
            tmp_val_sha[i] = Load(&values[nei_id1 * DIM + 4 * i]);
        }
        __syncthreads();
        for(unsigned j = i + 1; j < K; j ++){
            unsigned nei_id3 = nei0[j];
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                half4 val2 = Load(&values[nei_id3 * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }

            if(laneid == 0){
                float tmp_flt = __half2float(__hadd(val_res.x, val_res.y));
                if (tmp_flt < nei_distance[bid*K+j]){
                    atomicAdd(&detour_num[j], 1);
                }
            }
        }
        // next node
        for(unsigned i = laneid; i < (DIM / 4); i += blockDim.x){
            tmp_val_sha[i] = Load(&values[nei_id2 * DIM + 4 * i]);
        }
        __syncthreads();
        for(unsigned j = K - i; j < K; j ++){
            unsigned nei_id3 = nei0[j];
            half2 val_res;
            val_res.x = 0.0; val_res.y = 0.0;
            for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
                half4 val2 = Load(&values[nei_id3 * DIM + j * 4]);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
                val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
            }
            #pragma unroll
            for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
            }

            if(laneid == 0){
                float tmp_flt = __half2float(__hadd(val_res.x, val_res.y));
                if (tmp_flt < nei_distance[bid*K+j]){
                    atomicAdd(&detour_num[j], 1);
                }
            }
        }
    }
    __syncthreads();
    bitonic_sort_id_by_detour(detour_num, nei0, K);
    for(unsigned i = tid; i < d; i += blockDim.x * blockDim.y){
        unsigned tmp = atomicAdd(&reverse_num[nei0[i]], 1);
        if (tmp < K - d - 1) {
            reverse_graph[nei0[i] * d * 2 + tmp * 2] = bid;
            reverse_graph[nei0[i] * d * 2 + tmp * 2 + 1] = detour_num[i];
        }
        else{
            if(reverse_graph[nei0[i] * d * 2 + (tmp%d) * 2 + 1] > detour_num[i]){
                reverse_graph[nei0[i] * d * 2 + (tmp%d) * 2] = bid;
                reverse_graph[nei0[i] * d * 2 + (tmp%d) * 2 + 1] = detour_num[i];
            }
        }
        new_list[bid*K+i] = nei0[i];
    }
}

__global__ void select_1hop_cagra(unsigned* graph, unsigned d, unsigned* reverse_graph, unsigned K, unsigned* reverse_num, unsigned* new_list){
    __shared__ unsigned nei1[K_SIZE];
    __shared__ unsigned nei2[K_SIZE];
    __shared__ unsigned nei0[K_SIZE];
    __shared__ unsigned detour_num[K_SIZE];
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;

    collect_1hop(graph, K, nei0, detour_num, tid, bid);
    __syncthreads();
    filter_cagra(nei0, nei1, nei2, detour_num, graph, K, d, reverse_num, reverse_graph, new_list, tid, bid);
}

__global__ void select_1hop_dpg(unsigned* graph, unsigned d, unsigned* reverse_graph, unsigned K, unsigned* reverse_num, unsigned* new_list, unsigned DIM, const half* __restrict__ values, float* nei_distance){
    __shared__ unsigned nei1[K_SIZE];
    __shared__ unsigned nei2[K_SIZE];
    __shared__ unsigned nei0[K_SIZE];
    __shared__ unsigned detour_num[K_SIZE];
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;

    collect_1hop(graph, K, nei0, detour_num, tid, bid);
    __syncthreads();
    filter_dpg(nei0, nei1, nei2, detour_num, graph, K, d, reverse_num, reverse_graph, new_list, tid, bid, laneid, DIM, values, nei_distance);
}

__global__ void filter_reverse_1hop(unsigned* graph, unsigned* reverse_graph, unsigned d, unsigned K, unsigned* reverse_num, unsigned* new_list){
    __shared__ unsigned nei_id[K_SIZE];
    __shared__ unsigned reverse_nei[K_SIZE];
    __shared__ unsigned reverse_nei_detour[K_SIZE];
    __shared__ unsigned cur_num;
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;
    unsigned rever_num_cur = min(d, reverse_num[bid]);
    for(unsigned i = tid; i < rever_num_cur; i += blockDim.x * blockDim.y){
        reverse_nei[i] = reverse_graph[bid * d * 2 + i * 2];
        reverse_nei_detour[i] = reverse_graph[bid * d * 2 + i * 2 + 1];
    }
    for(unsigned i = tid; i < d; i += blockDim.x * blockDim.y){
        nei_id[i] = new_list[bid*K+i];
    }
    if(tid == 0) cur_num = 0;
    __syncthreads();

    bitonic_sort_id_by_detour(reverse_nei, reverse_nei_detour, rever_num_cur);
    for(unsigned i = tid + 1; i < rever_num_cur; i += blockDim.x * blockDim.y){
        if(reverse_nei[i] == reverse_nei[i - 1]){
            reverse_nei_detour[i] = 0xFFFFFFFF;
        }
    }
    for(unsigned i = tid; i < d/2; i += blockDim.x * blockDim.y){
        graph[bid * d + i] = nei_id[i];
    }
    __syncthreads();
    bitonic_sort_id_by_detour(reverse_nei_detour, reverse_nei, rever_num_cur);
    if(rever_num_cur < d / 2){
        for(unsigned i = tid; i < rever_num_cur; i += blockDim.x * blockDim.y){
            bool flag = false;
            for(unsigned j = 0; j < d - rever_num_cur; j++){
                if(reverse_nei[i] == nei_id[j]) {
                    flag = true;
                    break;
                }
            }
            if(!flag){
                unsigned tmp_id = atomicAdd(&cur_num, 1);
                graph[bid * d + d/2 + tmp_id] = reverse_nei[i];
            }
        }
        __syncthreads();
        for(unsigned i = tid; i < d/2 - cur_num; i += blockDim.x * blockDim.y){
            graph[bid * d + d/2 + cur_num + i] = nei_id[i + d/2];
        }
    }
    else{
        for(unsigned i = tid; cur_num < d/2 && i < rever_num_cur; i += blockDim.x * blockDim.y){
            bool flag = false;
            for(unsigned j = 0; j < d/2; j++){
                if(reverse_nei[i] == nei_id[j]) {
                    flag = true;
                    break;
                }
            }
            if(!flag){
                unsigned tmp_id = atomicAdd(&cur_num, 1);
                if(tmp_id < d/2)
                    graph[bid * d + d/2 + tmp_id] = reverse_nei[i];
            }
        }
        __syncthreads();
        for(unsigned i = tid; i < d/2 - min(cur_num, d/2); i += blockDim.x * blockDim.y){
            graph[bid * d + min(cur_num, d/2) + d/2 + i] = nei_id[i + d/2];
        }
    }
    __syncthreads();
    if(bid == 0 && tid == 0){
        for(unsigned i = 0; i < d; i++){
            printf("%d,", graph[bid * d + i]);
        }
        printf("\n");
    }
}

__global__ void cal_ep_gpu(unsigned* graph, unsigned* reverse_graph, unsigned* ep, const half* __restrict__ centers, const half* __restrict__ values, unsigned max_cand, unsigned DIM, unsigned TOPM, unsigned K){
    __shared__ unsigned shared_cand[SELECT_CAND+K_SIZE];
    __shared__ float shared_dis[SELECT_CAND+K_SIZE];
    __shared__ unsigned cur_nei;
    __shared__ unsigned tmp_id;
    __shared__ float cur_nei_dis;

    __shared__ unsigned top_M_Cand[TOPM_SIZE + K_SIZE];
    __shared__ unsigned top_M_Cand2[TOPM_SIZE + K_SIZE];
    __shared__ float top_M_Cand_dis[TOPM_SIZE + K_SIZE];

    __shared__ bool has_explore[TOPM_SIZE + K_SIZE];
    __shared__ half4 tmp_val_sha[DIM_SIZE / 4];
    __shared__ unsigned curr_of_candset, to_explore;
    unsigned tid = threadIdx.x + threadIdx.y * blockDim.x, bid = blockIdx.x, laneid = threadIdx.x;
    // initialize
    for(unsigned i = tid; i < K; i += blockDim.x * blockDim.y){
        shared_cand[K + tid] = graph[0 * K + tid];
        top_M_Cand[TOPM + tid] = graph[0 * K + tid];
        has_explore[TOPM + tid] = false;
    }
    if(tid == 0) {
        curr_of_candset = K;
    }
    __syncthreads();

    // half4 val1;
    // if(laneid < 24) val1 = Load(&centers[bid * DIM + 4 * laneid]);
    for(unsigned i = tid; i < (DIM / 4); i += blockDim.x * blockDim.y){
		tmp_val_sha[i] = Load(&centers[bid * DIM + 4 * i]);
	}
	__syncthreads();
    #pragma unroll
    for(unsigned i = threadIdx.y; i < K; i += blockDim.y){
        half2 val_res;
        val_res.x = 0.0; val_res.y = 0.0;
        // if(laneid < 24){
        //     half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + laneid * 4]);
        //     val_res = __hmul2(__hsub2(val1.x, val2.x), __hsub2(val1.x, val2.x));
        //     val_res = __hfma2(__hsub2(val1.y, val2.y), __hsub2(val1.y, val2.y), val_res);
        // }
        for(unsigned j = laneid; j < (DIM / 4); j += blockDim.x){
            half4 val2 = Load(&values[top_M_Cand[TOPM + i] * DIM + j * 4]);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].x, val2.x), __hsub2(tmp_val_sha[j].x, val2.x), val_res);
            val_res = __hfma2(__hsub2(tmp_val_sha[j].y, val2.y), __hsub2(tmp_val_sha[j].y, val2.y), val_res);
        }
        #pragma unroll
        for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
            val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
        }

        if(laneid == 0){
            top_M_Cand_dis[TOPM + i] = __half2float(__hadd(val_res.x, val_res.y));
            shared_dis[K + i] = top_M_Cand_dis[TOPM + i];
        }
    }
    __syncthreads();

    bitonic_sort_id_by_dis(&top_M_Cand_dis[TOPM], &top_M_Cand[TOPM], &has_explore[TOPM], K);
    for(unsigned i = tid; i < TOPM; i += blockDim.x * blockDim.y){
        top_M_Cand[i] = top_M_Cand[TOPM + i];
        top_M_Cand_dis[i] = top_M_Cand_dis[TOPM + i];
        has_explore[i] = has_explore[TOPM + i];
    }
    for(unsigned i = tid; i < TOPM + K; i += blockDim.x * blockDim.y){
        top_M_Cand2[i] = top_M_Cand[i];
    }

    __syncthreads();

    bitonic_sort_id_new2(top_M_Cand2, TOPM + K);
    // begin search
    for(unsigned i = 0; i < 30; i++){
        // explore the first node
        if(tid == 0){
            for(unsigned j = 0; j < TOPM; j++){
                if(has_explore[j] == false){
                    to_explore = top_M_Cand[j];
                    has_explore[j] = true;
                    break;
                }
                if(j == TOPM) {
                    to_explore = 0xFFFFFFFF;
                    // printf("AA %d\n", i);
                }
            }
        }
        __syncthreads();
        if(to_explore == 0xFFFFFFFF) {
            // if(bid == 0 && tid == 0) printf("AAA %d\n", i);
            break;
        }
        for(unsigned j = tid; j < K; j += blockDim.x * blockDim.y){
            unsigned to_append = graph[to_explore * K + j];
            // remove duplicate
            unsigned tmp = TOPM + K, res_id = 0;
            while (tmp > 1) {
                unsigned halfsize = tmp / 2;
                unsigned cand = top_M_Cand2[res_id + halfsize];
                res_id += ((cand <= to_append) ? halfsize : 0);
                tmp -= halfsize;
            }
            res_id += (top_M_Cand2[res_id] < to_append);
            if(res_id < K + TOPM && top_M_Cand2[res_id] == to_append){
                top_M_Cand_dis[TOPM + j] = INF_DIS;
            }
            else{
                top_M_Cand_dis[TOPM + j] = 0.0;
            }
            top_M_Cand[TOPM + j] = to_append;
            has_explore[TOPM + j] = false;
        }
        __syncthreads();
        // calculate distance
        #pragma unroll
        for(unsigned j = threadIdx.y; j < K; j += blockDim.y){
            if(top_M_Cand_dis[TOPM + j] < 1.0){
                
                half2 val_res;
                val_res.x = 0.0; val_res.y = 0.0;
                // if(laneid < 24){
                //     half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + laneid * 4]);
                //     val_res = __hmul2(__hsub2(val1.x, val2.x), __hsub2(val1.x, val2.x));
                //     val_res = __hfma2(__hsub2(val1.y, val2.y), __hsub2(val1.y, val2.y), val_res);
                // }
                for(unsigned k = laneid; k < (DIM / 4); k += blockDim.x){
                    half4 val2 = Load(&values[top_M_Cand[TOPM + j] * DIM + k * 4]);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].x, val2.x), __hsub2(tmp_val_sha[k].x, val2.x), val_res);
                    val_res = __hfma2(__hsub2(tmp_val_sha[k].y, val2.y), __hsub2(tmp_val_sha[k].y, val2.y), val_res);
                }
                
                #pragma unroll
                for(int lane_mask = 16; lane_mask > 0; lane_mask /= 2){  
                    val_res = __hadd2(val_res, __shfl_down_sync(0xffffffff, val_res, lane_mask));
                }

                if(laneid == 0){
                    top_M_Cand_dis[TOPM + j] = __half2float(__hadd(val_res.x, val_res.y));
                    unsigned tmp_res = atomicAdd(&curr_of_candset, 1);
                    // if(tmp_res < max_cand) {
                    //     candidates[bid][tmp_res] = top_M_Cand[TOPM + j];
                    //     cand_dis[bid][tmp_res] = top_M_Cand_dis[TOPM + j];
                    // }
                    shared_cand[K + tmp_res % max_cand] = top_M_Cand[TOPM + j];
                    shared_dis[K + tmp_res % max_cand] = top_M_Cand_dis[TOPM + j];
                    // if(bid == 836285 && top_M_Cand[TOPM + j] == 0) printf("CCCCC %.1f\n", top_M_Cand_dis[TOPM + j]);
                }
            }
        }
        __syncthreads();
        // merge into top_M
        bitonic_sort_id_by_dis(top_M_Cand_dis, top_M_Cand, has_explore, TOPM + K);
        for(unsigned i = tid; i < TOPM + K; i += blockDim.x * blockDim.y){
            top_M_Cand2[i] = top_M_Cand[i];
        }
        __syncthreads();

        bitonic_sort_id_new2(top_M_Cand2, TOPM + K);
    }
    if(tid == 0){
        *ep = top_M_Cand[0];
        // printf("ep: %d\n", *ep);
        // if(bid == 0){
        //     for(unsigned i = 0; i < K; i++){
        //         printf("(%d, %.1f); ", top_M_Cand[i], top_M_Cand_dis[i]);
        //     }
        //     printf("\n");
        // }
    }
}
