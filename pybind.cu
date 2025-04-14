#include <large_index.cuh>

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace nvcuda;

using namespace std;

namespace py = pybind11;

__device__ funcFormat dis_filter=distance_filter;
__device__ funcFormat ang_filter=angle_filter;

py::tuple GNN_descent(unsigned K, unsigned POINTS, unsigned DIM, unsigned iter, char* fname){
    float* data_load_float = NULL;
    unsigned points_num, dim;
    load_data(fname, data_load_float, points_num, dim);
    float tmp_ave = 0.0;
    for(unsigned i = 0; i < dim; i++)
        tmp_ave += abs(data_load_float[i]);
    tmp_ave /= dim;
    float norm_factor = 1.0;
    while(tmp_ave < 0.5){
        tmp_ave *= 10;
        norm_factor *= 10;
    }
    while(tmp_ave > 5){
        tmp_ave /= 10;
        norm_factor /= 10;
    }
    cout << "Points: " << points_num << ", Dim: " << dim << endl;
    
    float *center = new float [DIM];
    half *center_half = new half[DIM];
    for (unsigned j = 0; j < DIM; j++) center[j] = 0;
    for (unsigned i = 0; i < points_num; i++) {
        for (unsigned j = 0; j < DIM; j++) {
            center[j] += (data_load_float[i * DIM + j]);
        }
    }
    for (unsigned j = 0; j < DIM; j++) {
        center[j] /= (float)points_num;
        center_half[j] = __float2half(center[j]*norm_factor);
    }
    // float aa = compare(data_load_float + 123742 * DIM, center, DIM);
    // cout << "aaa: " << aa * aa << endl;
    half* center_dev;
    cudaMalloc((void**)&center_dev, dim * sizeof(half));
    cudaMemcpy(center_dev, center_half, sizeof(half) * dim, cudaMemcpyHostToDevice);

    half* data_half = new half[points_num * dim];
    for(unsigned i = 0; i < points_num * dim; i++){
        data_half[i] = __float2half(data_load_float[i]*norm_factor);
    }


    unsigned* graph_dev;
    cudaMalloc((void**)&graph_dev, points_num * K * sizeof(unsigned)); 
    unsigned* reverse_graph_dev;
    cudaMalloc((void**)&reverse_graph_dev, points_num * RESERVENUM * sizeof(unsigned));
    float* nei_distance;
    cudaMalloc((void**)&nei_distance, POINTS * K * sizeof(float));
    float* reverse_distance;
    cudaMalloc((void**)&reverse_distance, POINTS * RESERVENUM * sizeof(float));
    bool* nei_visit;
    cudaMalloc((void**)&nei_visit, POINTS * K * sizeof(bool));
    unsigned* reverse_num;
    cudaMalloc((void**)&reverse_num, POINTS * sizeof(unsigned));
    cudaMemset(reverse_num, 0, POINTS * sizeof(unsigned));
    unsigned* reverse_num_old;
    cudaMalloc((void**)&reverse_num_old, POINTS * sizeof(unsigned));
    cudaMemset(reverse_num_old, 0, POINTS * sizeof(unsigned));
    unsigned* new_num_global;
    cudaMalloc((void**)&new_num_global, POINTS * sizeof(unsigned));
    cudaMemset(new_num_global, 0, POINTS * sizeof(unsigned));
    unsigned* old_num_global;
    cudaMalloc((void**)&old_num_global, POINTS * sizeof(unsigned));
    cudaMemset(old_num_global, 0, POINTS * sizeof(unsigned));
    unsigned* hybrid_list;
    cudaMalloc((void**)&hybrid_list, POINTS * SAMPLE * 4 * sizeof(unsigned));
    half* data_dev;
    cudaMalloc((void**)&data_dev, points_num * dim * sizeof(half));
    cudaMemcpy(data_dev, data_half, points_num * dim * sizeof(half), cudaMemcpyHostToDevice);
    float* data_power_dev;
    cudaMalloc((void**)&data_power_dev, points_num * sizeof(float));
    cudaMemset(data_power_dev, 0, points_num * sizeof(float));

    dim3 grid(points_num, 1, 1);
    dim3 block(32, MAX_P / 32, 1);
    dim3 block2(32, 16, 1);
    dim3 block3(32, 3, 1);
    dim3 block4(32, 3, 1);

    unsigned all_it = iter/2, all_it2 = iter/2;
    auto start = std::chrono::high_resolution_clock::now();
    initialize_graph<<<points_num, 32>>>(graph_dev, points_num, nei_distance, nei_visit, K);

    cal_power<<<points_num, DIM>>>(data_dev, data_power_dev, dim, DIM);

    for(unsigned it = 0; it < all_it; it++){
        nn_descent_opt_sample<<<points_num, 32>>>(graph_dev, reverse_graph_dev, nei_visit ,reverse_num, reverse_num_old, new_num_global, old_num_global, hybrid_list, K);
        nn_descent_opt_reverse_sample<<<points_num, 32>>>(graph_dev, reverse_graph_dev, reverse_num, reverse_num_old, new_num_global, old_num_global, hybrid_list, K);
        reset_reverse_new_old_num<<<1000, 1024>>>(reverse_num,reverse_num_old,POINTS);
        nn_descent_opt_cal<<<grid, block2>>>(graph_dev, reverse_graph_dev, data_dev, data_power_dev, it, reverse_distance, reverse_num, new_num_global, old_num_global, hybrid_list, DIM, K);
        nn_descent_opt_merge<<<grid, block3>>>(graph_dev, reverse_graph_dev, it, all_it, nei_distance, reverse_distance, nei_visit, reverse_num, K);
        reset_reverse_num<<<1000, 1024>>>(reverse_num,POINTS);
    }
    do_reverse_graph<<<points_num, 32>>>(graph_dev, reverse_graph_dev, nei_distance, reverse_distance, reverse_num, K);
    reset_visit_reversenum<<<points_num, 32>>>(reverse_graph_dev, nei_visit, reverse_num, POINTS, K);
    for(unsigned it = 0; it < all_it2; it++){
        sample_kernel6<<<grid, block>>>(graph_dev, reverse_graph_dev, it, data_dev, points_num, all_it2, nei_distance, reverse_distance, nei_visit, reverse_num, DIM, K);
        if(it < all_it2 - 1) reset_reverse_num<<<1000, 1024>>>(reverse_num,POINTS);
    }
    merge_reverse_plus<<<grid, block3>>>(graph_dev, reverse_graph_dev, nei_distance, reverse_distance, reverse_num, K);
    reset_reverse_num<<<1000, 1024>>>(reverse_num, POINTS);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "time of GNN-Descent: " << duration.count() << "s" << std::endl;
    return py::make_tuple(
        py::capsule(graph_dev, [](void *ptr) { cudaFree(ptr); }),
        py::capsule(reverse_graph_dev, [](void *ptr) { cudaFree(ptr); }),
        py::capsule(data_dev, [](void *ptr) { cudaFree(ptr); }),
        py::capsule(nei_distance, [](void *ptr) { cudaFree(ptr); }),
        py::capsule(reverse_distance, [](void *ptr) { cudaFree(ptr); }),
        py::capsule(reverse_num, [](void *ptr) { cudaFree(ptr); }),
        py::capsule(center_dev, [](void *ptr) { cudaFree(ptr); }),
        py::capsule(hybrid_list, [](void *ptr) { cudaFree(ptr); })
    );
}

void Pruning(unsigned K, unsigned POINTS, unsigned DIM, unsigned FINAL_DEGREE, unsigned TOPM, py::tuple ptrs, string index_type, float thre=1.0, char* index_path="index.data"){

    // float* data_load_float = NULL;
    unsigned points_num=POINTS;
    // load_data(fname, data_load_float, points_num, dim);

    unsigned* graph_dev = static_cast<unsigned*>(ptrs[0].cast<py::capsule>().get_pointer());
    unsigned* reverse_graph_dev=static_cast<unsigned*>(ptrs[1].cast<py::capsule>().get_pointer());
    half* data_dev=static_cast<half*>(ptrs[2].cast<py::capsule>().get_pointer());
    float* nei_distance=static_cast<float*>(ptrs[3].cast<py::capsule>().get_pointer());
    float* reverse_distance=static_cast<float*>(ptrs[4].cast<py::capsule>().get_pointer());
    unsigned* reverse_num=static_cast<unsigned*>(ptrs[5].cast<py::capsule>().get_pointer());
    half* center_dev=static_cast<half*>(ptrs[6].cast<py::capsule>().get_pointer());
    unsigned* hybrid_list=static_cast<unsigned*>(ptrs[7].cast<py::capsule>().get_pointer());
    unsigned* ep_dev;
    cudaMalloc((void**)&ep_dev, sizeof(unsigned));
    // refine
    unsigned ep = 0;
    // vector<unsigned> res_graph(K * POINTS);
    // cal_ep(points_num, data_load_float, ep, res_graph.data(), TOPM, DIM, K);
    
    funcFormat dis_func, ang_func;
    cudaMemcpyFromSymbol(&dis_func,dis_filter,sizeof(funcFormat));
    cudaMemcpyFromSymbol(&ang_func,ang_filter,sizeof(funcFormat));

    vector<unsigned> res_graph(points_num * K);

    dim3 grid_s(points_num, 1, 1);
    dim3 block_s(32, 4, 1);

    dim3 grid_one(1, 1, 1);
    auto start = std::chrono::high_resolution_clock::now();
    if(index_type == "NSG"){
        cout << "begin to construct NSG" << endl;
        cal_ep_gpu<<<grid_one, block_s>>>(graph_dev, reverse_graph_dev, ep_dev, center_dev, data_dev, SELECT_CAND, DIM, TOPM, K);

        select_path<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, ep_dev, data_dev, SELECT_CAND, nei_distance, reverse_distance, reverse_num, DIM, FINAL_DEGREE, TOPM, K, 1.0);

        filter_reverse<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, data_dev, nei_distance, reverse_distance, reverse_num, DIM, FINAL_DEGREE, K, dis_func, 1.0);
        cudaMemcpy(res_graph.data(), graph_dev, points_num * K * sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ep, ep_dev, sizeof(unsigned), cudaMemcpyDeviceToHost);
        StoreNSG(index_path, res_graph.data(), FINAL_DEGREE, ep, points_num, K);
    }
    else if(index_type == "Vamana"){
        cout << "begin to construct Vamana" << endl;
        cal_ep_gpu<<<grid_one, block_s>>>(graph_dev, reverse_graph_dev, ep_dev, center_dev, data_dev, SELECT_CAND, DIM, TOPM, K);

        select_path<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, ep_dev, data_dev, SELECT_CAND, nei_distance, reverse_distance, reverse_num, DIM, FINAL_DEGREE, TOPM, K, thre);

        filter_reverse<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, data_dev, nei_distance, reverse_distance, reverse_num, DIM, FINAL_DEGREE, K, dis_func, thre);
        cudaMemcpy(res_graph.data(), graph_dev, points_num * K * sizeof(unsigned), cudaMemcpyDeviceToHost);
        cudaMemcpy(&ep, ep_dev, sizeof(unsigned), cudaMemcpyDeviceToHost);
        StoreVamana(index_path, res_graph.data(), ep, points_num, 0, K);
    }
    else if(index_type == "NSSG"){
        cout << "begin to construct NSSG" << endl;
        select_2hop<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, data_dev, SELECT_CAND, nei_distance, reverse_distance, reverse_num, DIM, FINAL_DEGREE, TOPM, K, thre);

        filter_reverse<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, data_dev, nei_distance, reverse_distance, reverse_num, DIM, FINAL_DEGREE, K, ang_func, thre);
        cudaMemcpy(res_graph.data(), graph_dev, points_num * K * sizeof(unsigned), cudaMemcpyDeviceToHost);
        StoreNSSG(index_path, res_graph.data(), FINAL_DEGREE, points_num, K);
    }

    else if(index_type == "CAGRA"){
        cout << "begin to construct CAGRA" << endl;
        select_1hop_cagra<<<grid_s, block_s>>>(graph_dev, FINAL_DEGREE, reverse_graph_dev, K, reverse_num, hybrid_list);
        filter_reverse_1hop<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, FINAL_DEGREE, K, reverse_num, hybrid_list);
        cudaMemcpy(res_graph.data(), graph_dev, points_num * K * sizeof(unsigned), cudaMemcpyDeviceToHost);
        StoreCAGRA(index_path, res_graph.data(), points_num, FINAL_DEGREE, K);
    }

    else if(index_type == "DPG"){
        cout << "begin to construct DPG" << endl;
        select_1hop_dpg<<<grid_s, block_s>>>(graph_dev, FINAL_DEGREE, reverse_graph_dev, K, reverse_num, hybrid_list, DIM, data_dev, nei_distance);
        filter_reverse_1hop<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, FINAL_DEGREE, K, reverse_num, hybrid_list);
        cudaMemcpy(res_graph.data(), graph_dev, points_num * K * sizeof(unsigned), cudaMemcpyDeviceToHost);
        StoreDPG(index_path, res_graph.data(), points_num, FINAL_DEGREE, K);
    }

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "time of pruning: " << duration.count() << "s" << std::endl;
}

void Pruning_with_CFS(unsigned K, unsigned POINTS, unsigned DIM, unsigned FINAL_DEGREE, unsigned TOPM, py::tuple ptrs, string mode, string metric, float thre){

    // float* data_load_float = NULL;
    unsigned points_num=POINTS;
    // load_data(fname, data_load_float, points_num, dim);

    unsigned* graph_dev = static_cast<unsigned*>(ptrs[0].cast<py::capsule>().get_pointer());
    unsigned* reverse_graph_dev=static_cast<unsigned*>(ptrs[1].cast<py::capsule>().get_pointer());
    half* data_dev=static_cast<half*>(ptrs[2].cast<py::capsule>().get_pointer());
    float* nei_distance=static_cast<float*>(ptrs[3].cast<py::capsule>().get_pointer());
    float* reverse_distance=static_cast<float*>(ptrs[4].cast<py::capsule>().get_pointer());
    unsigned* reverse_num=static_cast<unsigned*>(ptrs[5].cast<py::capsule>().get_pointer());
    half* center_dev=static_cast<half*>(ptrs[6].cast<py::capsule>().get_pointer());
    unsigned* hybrid_list=static_cast<unsigned*>(ptrs[7].cast<py::capsule>().get_pointer());
    unsigned* ep_dev;
    cudaMalloc((void**)&ep_dev, sizeof(unsigned));
    // refine
    unsigned ep = 0;
    // vector<unsigned> res_graph(K * POINTS);
    // cal_ep(points_num, data_load_float, ep, res_graph.data(), TOPM, DIM, K);
    
    funcFormat dis_func, ang_func;
    cudaMemcpyFromSymbol(&dis_func,dis_filter,sizeof(funcFormat));
    cudaMemcpyFromSymbol(&ang_func,ang_filter,sizeof(funcFormat));

    vector<unsigned> res_graph(points_num * K);

    dim3 grid_s(points_num, 1, 1);
    dim3 block_s(32, 4, 1);

    dim3 grid_one(1, 1, 1);
    auto start = std::chrono::high_resolution_clock::now();
    if(mode == "path"){
        cal_ep_gpu<<<grid_one, block_s>>>(graph_dev, reverse_graph_dev, ep_dev, center_dev, data_dev, SELECT_CAND, DIM, TOPM, K);
        select_path<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, ep_dev, data_dev, SELECT_CAND, nei_distance, reverse_distance, reverse_num, DIM, FINAL_DEGREE, TOPM, K, thre);
    }
    else if(mode == "2-hop"){
        select_2hop<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, data_dev, SELECT_CAND, nei_distance, reverse_distance, reverse_num, DIM, FINAL_DEGREE, TOPM, K, thre);
    }

    else if(mode == "1-hop"){
        select_1hop_cagra<<<grid_s, block_s>>>(graph_dev, FINAL_DEGREE, reverse_graph_dev, K, reverse_num, hybrid_list);
    }

    if(metric == "dist"){
        filter_reverse<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, data_dev, nei_distance, reverse_distance, reverse_num, DIM, FINAL_DEGREE, K, dis_func, thre);
    }
    else if(metric == "angle"){
        filter_reverse<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, data_dev, nei_distance, reverse_distance, reverse_num, DIM, FINAL_DEGREE, K, ang_func, thre);
    }
    else if(metric == "rank"){
        filter_reverse_1hop<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, FINAL_DEGREE, K, reverse_num, hybrid_list);
    }

    cudaMemcpy(res_graph.data(), graph_dev, points_num * K * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ep, ep_dev, sizeof(unsigned), cudaMemcpyDeviceToHost);
    // StoreNSG(index_path, res_graph.data(), FINAL_DEGREE, ep, points_num, K);

    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "time of pruning: " << duration.count() << "s" << std::endl;
}

void largeIndex(vector<unsigned>& devicelist, unsigned DIM, unsigned K, unsigned cluster_num, unsigned max_points, string file_prefix, string local_index_path_prefix, string index_store_path, unsigned max_degree, unsigned buffer_size, unsigned all_node_num, char* centers_path, float thre, unsigned FINAL_DEGREE, unsigned TOPM){
    unsigned dim = DIM;
    vector<vector<unsigned>> graph_pool(buffer_size);
    vector<bool> has_built(cluster_num), has_merged(cluster_num);
    funcFormat dis_func;
    for(unsigned i = 0; i < devicelist.size(); i++){
        cudaSetDevice(devicelist[i]);
        cudaMemcpyFromSymbol(&dis_func,dis_filter,sizeof(funcFormat));
    }

    // initialize graph_pool;
    for(unsigned i = 0; i < buffer_size; i++){
        graph_pool[i].resize(max_points * K);
    }
    cout << "End init graph pool" << endl;

    // process cluster
    vector<unsigned> Centers, Centers_second, results;
    std::ifstream in_c(centers_path, std::ios::binary);
    unsigned center, second_center;
    if(!in_c.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    for(unsigned i = 0; i < all_node_num; i++){
        in_c.read((char*)&center, 4);
        in_c.read((char*)&second_center, 4);
        Centers.push_back(center);
        Centers_second.push_back(second_center);
    }
    vector<vector<unsigned>> clusters(cluster_num);
    cout << "End read" << endl;
    for(unsigned i = 0; i < all_node_num; i++){
        clusters[Centers[i]].push_back(i);
        clusters[Centers_second[i]].push_back(i);
    }
    // adjust the clusters
    unsigned tmp_cluster_num = 0;
    for(unsigned i = 0; i < cluster_num; i++){
        if(clusters[i].size() > max_points){
            unsigned div_factor = 2;
            while(clusters[i].size() / div_factor > max_points) div_factor++;
            vector<unsigned> cand_cluster_id = {i};
            for(unsigned j = 0; j < div_factor - 1; j++){
                cand_cluster_id.push_back(cluster_num + tmp_cluster_num + j);
            }
            for(unsigned j = 0; j < clusters[i].size(); j++){
                if(Centers[clusters[i][j]] == i) Centers[clusters[i][j]] = cand_cluster_id[j % div_factor];
                else Centers_second[clusters[i][j]] = cand_cluster_id[j % div_factor];
            }
            tmp_cluster_num += (div_factor - 1);
        }
    }
    cluster_num += tmp_cluster_num;
    clusters.clear();
    clusters.resize(cluster_num);
    for(unsigned i = 0; i < all_node_num; i++){
        clusters[Centers[i]].push_back(i);
        clusters[Centers_second[i]].push_back(i);
    }
    cout << "Cluster num after being adjusted:" << cluster_num << endl; 
    // for(unsigned i = 0; i < cluster_num; i++){
    //     cout << clusters[i].size() << ",";
    // }
    // cout << endl;

    vector<unsigned> position(cluster_num);
    has_built.resize(cluster_num);
    has_merged.resize(cluster_num);
    for(unsigned i = 0; i < cluster_num; i++){
        has_built[i] = false;
        has_merged[i] = false;
    }

    order_merge_main(Centers, Centers_second, cluster_num, all_node_num, results, clusters, position, buffer_size);

    vector<unsigned> metroids(cluster_num);
    bool lock_global = false;
    auto start = std::chrono::high_resolution_clock::now();
    
    thread t1(order_merge_new, ref(clusters), all_node_num, cluster_num, ref(Centers), ref(Centers_second), ref(results), ref(position), ref(metroids), index_store_path, local_index_path_prefix, K, ref(graph_pool), ref(has_merged), ref(has_built), max_degree, buffer_size);
    vector<thread> construct_threads;
    for(unsigned i = 0; i < devicelist.size(); i++){
        construct_threads.push_back(thread(gpu_construct, ref(clusters), cluster_num, ref(results), ref(position), ref(metroids), ref(devicelist), i, file_prefix, DIM, max_points, K, ref(graph_pool), ref(has_merged), ref(has_built), thre, FINAL_DEGREE, TOPM, dis_func));
    }
    
    t1.join();
    for(unsigned i = 0; i < devicelist.size(); i++){
        construct_threads[i].join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "time for indexing: " << duration.count() << "s" << std::endl;
}
PYBIND11_MODULE(Tagore, m) {
    m.doc() = "Tagore GPU Indexing";
    m.def("GNN_descent", &GNN_descent, "GNN_descent");
    m.def("Pruning", &Pruning, "Pruning", py::arg("K"), py::arg("POINTS"), py::arg("DIM"), py::arg("FINAL_DEGREE"), py::arg("TOPM")=64, py::arg("ptrs"), py::arg("index_type"), py::arg("thre")=1.0, py::arg("index_path")="index.data");
    m.def("Pruning_with_CFS", &Pruning_with_CFS, "Pruning_with_CFS");
    m.def("largeIndex", &largeIndex, "largeIndex");
}
