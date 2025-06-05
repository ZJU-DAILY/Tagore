#include <large_index.cuh>

using namespace std;

/*
Algorithm 4 in the paper: cluster-aware dispatching strategy
This function reorders the clusters based on their overlapped nodes
*/
void reorder(vector<map<unsigned, unsigned>>& intersect, unsigned cluster_num, unsigned buffer_size, unsigned ini, vector<unsigned>& result, vector<unsigned>& position){
    vector<unsigned> buffer;
    vector<bool> visited(cluster_num);
    unsigned init_node = ini;
    result.push_back(init_node);
    visited[init_node] = true;
    buffer.push_back(init_node);
    position[init_node] = 0;
    unsigned all_sum = 0;
    for(unsigned i = 1; i < buffer_size; i++){
        // cout << i << endl;
        unsigned max_sum = 0, max_id;
        for(unsigned j = 0; j < cluster_num; j++){
            if(visited[j]) continue;
            unsigned tmp_sum = 0;
            for(unsigned k = 0; k < result.size(); k++){
                auto it = intersect[j].find(result[k]);
                if(it != intersect[j].end()){
                    tmp_sum += it->second;
                }
            }
            if(tmp_sum > max_sum){
                max_sum = tmp_sum;
                max_id = j;
            }
        }
        result.push_back(max_id);
        buffer.push_back(max_id);
        position[max_id] = i;
        visited[max_id] = true;
        all_sum += max_sum;
    }
    // cout << "End init" << endl; 
    // omp_lock_t lock;
    // omp_init_lock(&lock);
    for(unsigned i = 0; i < cluster_num - buffer_size; i++){
        cout << i + buffer_size << endl;
        unsigned max_sum = 0, max_id, max_evict;
        // #pragma omp parallel for
        for(unsigned j = 0; j < cluster_num; j++){
            if(visited[j]) continue;
            unsigned max_local_evict, max_local_sum = 0;
            for(unsigned k = 0; k < buffer_size; k++){
                unsigned tmp_sum = 0;
                for(unsigned l = k + 1; (l % buffer_size) != k; l++){
                    auto it = intersect[j].find(buffer[l % buffer_size]);
                    if(it != intersect[j].end()){
                        tmp_sum += it->second;
                    }
                }
                if(tmp_sum > max_local_sum){
                    max_local_sum = tmp_sum;
                    max_local_evict = k;
                }
            }
            // omp_set_lock(&lock);
            if(max_local_sum > max_sum){
                max_sum = max_local_sum;
                max_id = j;
                max_evict = max_local_evict;
            }
            // omp_unset_lock(&lock);
        }
        
        buffer[max_evict] = max_id;
        result.push_back(max_id);
        visited[max_id] = true;
        all_sum += max_sum;
        position[max_id] = max_evict;
        // cout << max_sum << "," << max_id << "," << max_evict << ";";
        // cout << max_evict << ",";
    }
    // omp_destroy_lock(&lock);
    // cout << endl;
}

// prepare the data for reordering
void order_merge_main(vector<unsigned>& Centers, vector<unsigned>& Centers_second, unsigned cluster_num, unsigned points_num, vector<unsigned>& result, vector<vector<unsigned>>& clusters, vector<unsigned>& position, unsigned buffer_size){
    vector<map<unsigned, unsigned>> intersect(cluster_num);
    for(unsigned i = 0; i < points_num; i++){
        auto it = intersect[Centers[i]].find(Centers_second[i]);
        if(it != intersect[Centers[i]].end()){
            it->second++;
        }
        else{
            intersect[Centers[i]][Centers_second[i]] = 1;
        }

        it = intersect[Centers_second[i]].find(Centers[i]);
        if(it != intersect[Centers_second[i]].end()){
            it->second++;
        }
        else{
            intersect[Centers_second[i]][Centers[i]] = 1;
        }
    }
    reorder(intersect, cluster_num, buffer_size, 0, result, position);
}

// convert float type to half type for central point
__global__ void f2h_center(float* data, half* data_half, unsigned DIM){
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(unsigned i = tid; i < DIM; i += blockDim.x * gridDim.x){
        data_half[i] = __float2half(data[i]);
    }
}

// convert float type to half type
__global__ void f2h(float* data, half* data_half, unsigned POINTS, unsigned DIM){
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    for(unsigned i = tid; i < POINTS * DIM; i += blockDim.x * gridDim.x){
        data_half[i] = __float2half(data[i]);
    }
}

/*
Subgraph construction function using GPUs
This function performs GNN-Descent and CFS pruning sequentially. 
Before that, it prepares the required data during the construction. 
Finally, it stores the subgraphs into the buffer on the CPU. 
*/
void gpu_construct(vector<vector<unsigned>>& clusters, unsigned cluster_num, vector<unsigned>& order, vector<unsigned>& graph_position, vector<unsigned>& metroids, vector<unsigned>& devicelist, unsigned deviceID, string file_prefix, unsigned DIM, unsigned max_points, unsigned K, vector<vector<unsigned>>& graph_pool, vector<bool>& has_merged, vector<bool>& has_built, float thre, unsigned FINAL_DEGREE, unsigned TOPM, funcFormat dis_filter_new){
    unsigned dim = DIM, points_num;
    unsigned* graph_dev;
    unsigned* reverse_graph_dev;
    float* data_dev;
    float* data_power_dev;
    unsigned* ep_dev;
    float* center_dev;
    unsigned ep = 0;
    half* data_half_dev, *center_dev_half;

    cudaSetDevice(devicelist[deviceID]);

    cudaMalloc((void**)&data_dev, max_points * dim * sizeof(float));
    cudaMalloc((void**)&reverse_graph_dev, max_points * RESERVENUM * sizeof(unsigned));
    cudaMalloc((void**)&graph_dev, max_points * K * sizeof(unsigned));
    cudaMalloc((void**)&data_power_dev, max_points * sizeof(float));
    cudaMalloc((void**)&ep_dev, sizeof(unsigned));
    cudaMalloc((void**)&center_dev, dim * sizeof(float));

    cudaMalloc((void**)&data_half_dev, max_points * dim * sizeof(half));
    cudaMalloc((void**)&center_dev_half, dim * sizeof(half));

    float* nei_distance;
    cudaMalloc((void**)&nei_distance, max_points * K * sizeof(float));
    float* reverse_distance;
    cudaMalloc((void**)&reverse_distance, max_points * RESERVENUM * sizeof(float));
    bool* nei_visit;
    cudaMalloc((void**)&nei_visit, max_points * K * sizeof(bool));
    unsigned* reverse_num;
    cudaMalloc((void**)&reverse_num, max_points * sizeof(unsigned));
    cudaMemset(reverse_num, 0, max_points * sizeof(unsigned));
    unsigned* reverse_num_old;
    cudaMalloc((void**)&reverse_num_old, max_points * sizeof(unsigned));
    cudaMemset(reverse_num_old, 0, max_points * sizeof(unsigned));
    unsigned* new_num_global;
    cudaMalloc((void**)&new_num_global, max_points * sizeof(unsigned));
    cudaMemset(new_num_global, 0, max_points * sizeof(unsigned));
    unsigned* old_num_global;
    cudaMalloc((void**)&old_num_global, max_points * sizeof(unsigned));
    cudaMemset(old_num_global, 0, max_points * sizeof(unsigned));
    unsigned* hybrid_list;
    cudaMalloc((void**)&hybrid_list, max_points * SAMPLE * 4 * sizeof(unsigned));
    

    for(unsigned cid = deviceID; cid < cluster_num; cid+=devicelist.size()){
        
        float* data = NULL;
        data = new float[clusters[order[cid]].size() * dim];
        points_num = clusters[order[cid]].size();
        cout << "Begin to construct batch" << endl;

        std::string file_path = file_prefix + std::to_string(order[cid]);
        std::ifstream in(file_path, std::ios::binary);
        unsigned tmp_points;
        in.read((char*)&tmp_points, sizeof(unsigned));

        in.read((char*)data, ((size_t)points_num) * ((size_t)dim) * sizeof(float));
        in.close();

        cudaMemcpy(data_dev, data, points_num * dim * sizeof(float), cudaMemcpyHostToDevice);

        cout << "Device:" << devicelist[deviceID] << "; Points num:" << points_num << " has been move to GPU" << endl;
        cudaMemset(data_power_dev, 0, points_num * sizeof(float));

        float *center = new float [DIM];
        for (unsigned j = 0; j < DIM; j++) center[j] = 0;
        for (unsigned i = 0; i < points_num; i++) {
            for (unsigned j = 0; j < DIM; j++) {
                center[j] += data[i * DIM + j];
            }
        }
        for (unsigned j = 0; j < DIM; j++) {
            center[j] /= (float)points_num;
        }
        
        cudaMemcpy(center_dev, center, sizeof(float) * dim, cudaMemcpyHostToDevice);

        delete[] data;
        delete[] center;

        f2h_center<<<1, DIM>>>(center_dev, center_dev_half, DIM);
        dim3 grid(points_num, 1, 1);
        dim3 block(32, MAX_P / 32, 1);
        dim3 block2(32, 16, 1);
        dim3 block3(32, 3, 1);
        dim3 block4(32, 3, 1);

        dim3 grid_s(points_num, 1, 1);
        dim3 block_s(32, 3, 1);
        dim3 grid_one(1, 1, 1);

        funcFormat dis_func;
        cudaMemcpyFromSymbol(&dis_func,dis_filter_new,sizeof(funcFormat));

        unsigned all_it = 8, all_it2 = 8;

        initialize_graph<<<points_num, 32>>>(graph_dev, points_num, nei_distance, nei_visit, K);
        f2h<<<points_num, DIM>>>(data_dev, data_half_dev, points_num, dim);
        cal_power<<<points_num, dim>>>(data_half_dev, data_power_dev, dim, DIM);

        for(unsigned it = 0; it < all_it; it++){
            nn_descent_opt_sample<<<points_num, 32>>>(graph_dev, reverse_graph_dev, nei_visit ,reverse_num, reverse_num_old, new_num_global, old_num_global, hybrid_list, K);
            nn_descent_opt_reverse_sample<<<points_num, 32>>>(graph_dev, reverse_graph_dev, reverse_num, reverse_num_old, new_num_global, old_num_global, hybrid_list, K);
            reset_reverse_new_old_num<<<1000, 1024>>>(reverse_num,reverse_num_old,max_points);
            nn_descent_opt_cal<<<grid, block2>>>(graph_dev, reverse_graph_dev, data_half_dev, data_power_dev, it, reverse_distance, reverse_num, new_num_global, old_num_global, hybrid_list, DIM, K);
            nn_descent_opt_merge<<<grid, block3>>>(graph_dev, reverse_graph_dev, it, all_it, nei_distance, reverse_distance, nei_visit, reverse_num, K);
            reset_reverse_num<<<1000, 1024>>>(reverse_num,max_points);
        }
        do_reverse_graph<<<points_num, 32>>>(graph_dev, reverse_graph_dev, nei_distance, reverse_distance, reverse_num, K);
        reset_visit_reversenum<<<points_num, 32>>>(reverse_graph_dev, nei_visit, reverse_num, max_points, K);
        for(unsigned it = 0; it < all_it2; it++){
            sample_kernel6<<<grid, block>>>(graph_dev, reverse_graph_dev, it, data_half_dev, points_num, all_it2, nei_distance, reverse_distance, nei_visit, reverse_num, DIM, K);
        if(it < all_it2 - 1) reset_reverse_num<<<1000, 1024>>>(reverse_num,max_points);
        }
        merge_reverse_plus<<<grid, block3>>>(graph_dev, reverse_graph_dev, nei_distance, reverse_distance, reverse_num, K);
        reset_reverse_num<<<1000, 1024>>>(reverse_num, max_points);

        cal_ep_gpu<<<grid_one, block_s>>>(graph_dev, reverse_graph_dev, ep_dev, center_dev_half, data_half_dev, SELECT_CAND, DIM, TOPM, K);

        select_path<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, ep_dev, data_half_dev, SELECT_CAND, nei_distance, reverse_distance, reverse_num, DIM, FINAL_DEGREE, TOPM, K, thre);

        filter_reverse<<<grid_s, block_s>>>(graph_dev, reverse_graph_dev, data_half_dev, nei_distance, reverse_distance, reverse_num, DIM, FINAL_DEGREE, K, dis_func, thre);
        reset_reverse_num<<<1000, 1024>>>(reverse_num, max_points);
        cudaMemcpy(&ep, ep_dev, sizeof(unsigned), cudaMemcpyDeviceToHost);
    
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        cout << "Device:" << devicelist[deviceID] << "; ep:" << clusters[order[cid]][ep] << endl;
        metroids[cid] = clusters[order[cid]][ep];

        cout << "Device:" << devicelist[deviceID] << "; Waiting for transfering No." << cid << " to CPU" << endl;
        while(has_merged[order[cid]]==false) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        cudaMemcpy(graph_pool[graph_position[order[cid]]].data(), graph_dev, points_num * K * sizeof(unsigned), cudaMemcpyDeviceToHost);
        
        cout << "Device:" << devicelist[deviceID] << "; No." << cid << " has been transfered to CPU" << endl;
        for(unsigned i = 0; i < points_num; i++){
            unsigned degree = graph_pool[graph_position[order[cid]]][i * K];
            for(unsigned j = 1; j <= degree; j++){
                graph_pool[graph_position[order[cid]]][i * K + j] = clusters[order[cid]][graph_pool[graph_position[order[cid]]][i * K + j]];
            }
        }
        has_built[order[cid]] = true;
        
        cout << "Device:" << devicelist[deviceID] << "; No." << cid << " has been mapped" << endl;
    }
}

/* 
Algorithm 3 in the paper: merge the subgraphs.
It asynchronously merges the subgraphs using the CPU. 
Finally, it stores the global graph index into the disk. 
*/
void order_merge_new(vector<vector<unsigned>>& clusters, unsigned points_num, unsigned cluster_num, vector<unsigned>& Centers, vector<unsigned>& Centers_second, vector<unsigned>& order, vector<unsigned>& graph_position, vector<unsigned>& metroids, string index_store_path, string local_index_path_prefix, unsigned K, vector<vector<unsigned>>& graph_pool, vector<bool>& has_merged, vector<bool>& has_built, unsigned max_degree, unsigned buffer_size){

    uint64_t index_size = 0;
    vector<unsigned> cluster_map(cluster_num);
    

    for(unsigned i = 0; i < order.size(); i++){
        cluster_map[order[i]] = i;
    }
    for(unsigned i = 0; i < buffer_size; i++){
        has_merged[order[i]] = true;
    }
    vector<unsigned> buffer(buffer_size, 0xFFFFFFFF);


    vector<unsigned> pointers(points_num);
    vector<bool> visited(points_num), node_set(points_num), cached(points_num);
    cout << "Waiting to merge No.0" << endl;
    while(has_built[order[0]] == false) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    cout << "Begin to merge No.0" << endl;
    for(unsigned i = 0; i < clusters[order[0]].size(); i++){
        visited[clusters[order[0]][i]] = true;
        pointers[clusters[order[0]][i]] = i;
    }
    buffer[0] = order[0];
    cout << "End to merge No.0" << endl;
    unsigned cache_num = 0, uncache_num = 0;
    
    try{
        for(unsigned i = 1; i < cluster_num; i++){
            std::random_device rng;
            std::mt19937 urng(rng());
            cout << "Waiting to merge No." << i << endl;
            while(has_built[order[i]] == false){
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            cout << "Begin to merge No." << i << endl;

            vector<unsigned> nhood;

            for(unsigned j = 0; j < clusters[order[i]].size(); j++){
                unsigned node = clusters[order[i]][j];
                if(visited[node]){
                    if(pointers[node] != 0xFFFFFFFF){
                        cache_num++;
                        unsigned ano_clu = (Centers[node] == order[i] ? Centers_second[node] : Centers[node]);
                        if(Centers[node] != order[i]){
                            unsigned degree = graph_pool[graph_position[ano_clu]][pointers[node] * K];
                            for(unsigned k = 1; k <= degree; k++){
                                unsigned tmp_node = graph_pool[graph_position[ano_clu]][pointers[node] * K + k];
                                // if(node_set[tmp_node] == 0){
                                    // nhood.push_back(tmp_node);
                                    node_set[tmp_node] = 1;
                                    nhood.push_back(tmp_node);
                                // }
                            }

                            unsigned new_degree = graph_pool[graph_position[order[i]]][j * K];
                            for(unsigned k = 1; k <= new_degree; k++){
                                unsigned node_tmp = graph_pool[graph_position[order[i]]][j * K + k];
                                if(node_set[node_tmp] == 0){
                                    node_set[node_tmp] = 1;
                                    nhood.push_back(node_tmp);
                                }
                            }
                        }
                        else{
                            unsigned degree = graph_pool[graph_position[order[i]]][j * K];
                            for(unsigned k = 1; k <= degree; k++){
                                unsigned tmp_node = graph_pool[graph_position[order[i]]][j * K + k];
                                node_set[tmp_node] = 1;
                                nhood.push_back(tmp_node);
                            }

                            unsigned new_degree = graph_pool[graph_position[ano_clu]][pointers[node] * K];
                            for(unsigned k = 1; k <= new_degree; k++){
                                unsigned node_tmp = graph_pool[graph_position[ano_clu]][pointers[node] * K + k];
                                if(node_set[node_tmp] == 0){
                                    node_set[node_tmp] = 1;
                                    nhood.push_back(node_tmp);
                                }
                            }
                        }
                        // std::shuffle(nhood.begin(), nhood.end(), urng);
                        for(unsigned k = 1; k <= min((unsigned)nhood.size(), max_degree); k++){
                            graph_pool[graph_position[ano_clu]][pointers[node] * K + k] = nhood[k - 1];
                        }
                        graph_pool[graph_position[ano_clu]][pointers[node] * K] = min((unsigned)nhood.size(), max_degree);
                        // if(count + degree > max_degree) cout << "B";
                        graph_pool[graph_position[order[i]]][j * K] = 0;
                        for(unsigned ll = 1; ll <= nhood.size(); ll++){
                            node_set[nhood[ll - 1]] = 0;
                        }
                        nhood.clear();
                        cached[node] = true;
                    }
                }
                else{
                    visited[node] = true;
                    pointers[node] = j;
                }
            }
            buffer[graph_position[order[i]]] = order[i];
            // cout << "A";
            if(i >= buffer_size - 1 && i + 1 < cluster_num){
                unsigned clu_id = buffer[graph_position[order[i + 1]]];
                if(clu_id != 0xFFFFFFFF){
                    std::string file_path = local_index_path_prefix + std::to_string(clu_id);

                    cached_ofstream out(file_path, BUFFER_SIZE_FOR_CACHED_READ);
                    for(unsigned j = 0; j < clusters[clu_id].size(); j++){
                        unsigned node = clusters[clu_id][j];
                        unsigned degree = graph_pool[graph_position[clu_id]][j * K];
                        out.write((char *)&degree, sizeof(unsigned));
                        out.write((char *)(&graph_pool[graph_position[clu_id]][j * K + 1]), sizeof(unsigned) * degree);
                        visited[node] = true;
                        pointers[node] = 0xFFFFFFFF;
                        
                    }
                    out.close();
                }

                has_merged[order[i + 1]] = true;
            }
            cout << "Buffer:" ;
            for(unsigned l=0;l<buffer_size;l++){
                cout << buffer[l] << ",";
            }
            cout << endl;


            cout << "End to merge No." << i << endl;
        }
        for(unsigned i = 0; i < buffer_size; i++){
            unsigned clu_id = buffer[i];
            if(clu_id != 0xFFFFFFFF){
                std::string file_path = local_index_path_prefix + std::to_string(clu_id);

                cached_ofstream out(file_path, BUFFER_SIZE_FOR_CACHED_READ);
                for(unsigned j = 0; j < clusters[clu_id].size(); j++){
                    unsigned node = clusters[clu_id][j];
                    unsigned degree = graph_pool[graph_position[clu_id]][j * K];
                    out.write((char *)&degree, sizeof(unsigned));
                    if(degree > 0) out.write((char *)(&graph_pool[graph_position[clu_id]][j * K + 1]), sizeof(unsigned) * degree);
                    visited[node] = true;
                    pointers[node] = 0xFFFFFFFF;
                }
                out.close();
            }
        }
    }
    catch (const std::runtime_error& e) {
        std::cerr << "File Exception caught: " << e.what() << __FILE__ << __FUNCTION__ << __LINE__ << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "File Some other exception caught: " << e.what() << __FILE__ << __FUNCTION__ << __LINE__ << std::endl;
    }

    
    cout << cache_num << "," << uncache_num << endl;

    cout << "Begin to merge" << endl;
    cached_ofstream out_merge(index_store_path, BUFFER_SIZE_FOR_CACHED_IO);

    
    out_merge.write((char*)&index_size, sizeof(uint64_t));
    out_merge.write((char*)&max_degree, sizeof(unsigned));
    out_merge.write((char*)&max_degree, sizeof(unsigned));
    out_merge.write((char*)&index_size, sizeof(uint64_t));

    vector<cached_ifstream> shard_index(cluster_num);
    for(unsigned i = 0; i < cluster_num; i++){
        std::string file_path = local_index_path_prefix + std::to_string(i);
        shard_index[i].open(file_path, BUFFER_SIZE_FOR_CACHED_READ);
    }

    vector<unsigned> neighs(2 * max_degree), neighbors(max_degree), nhood;

    for(unsigned i = 0; i < points_num; i++){
        if(cached[i]){
            unsigned clu_id = Centers[i];
            unsigned degree, tmp_degree;
            shard_index[clu_id].read((char*)&degree, sizeof(unsigned));
            if(degree == 0){
                clu_id = Centers_second[i];
                shard_index[clu_id].read((char*)&degree, sizeof(unsigned));
            }
            else{
                shard_index[Centers_second[i]].read((char*)&tmp_degree, sizeof(unsigned));
            }
            shard_index[clu_id].read((char*)neighs.data(), degree * sizeof(unsigned));
            out_merge.write((char*)&degree, sizeof(unsigned));
            out_merge.write((char*)neighs.data(), degree * sizeof(unsigned));
            index_size += (uint64_t)((1 + degree) * sizeof(unsigned));
        }
        else{
            unsigned degree1, degree2, count = 0;
            shard_index[Centers[i]].read((char*)&degree1, sizeof(unsigned));
            shard_index[Centers[i]].read((char*)neighs.data(), degree1 * sizeof(unsigned));
            shard_index[Centers_second[i]].read((char*)&degree2, sizeof(unsigned));
            shard_index[Centers_second[i]].read((char*)neighbors.data(), degree2 * sizeof(unsigned));
            for(unsigned j = 0; j < degree1; j++){
                node_set[neighs[j]] = 1;
                nhood.push_back(neighs[j]);
            }
            for(unsigned j = 0; j < degree2; j++){
                if(node_set[neighbors[j]] == 0){
                    neighs[degree1 + j] = neighbors[j];
                    node_set[neighbors[j]] = 1;
                    nhood.push_back(neighbors[j]);
                    count++;
                }
            }
            for(unsigned j = 0; j < nhood.size(); j++){
                node_set[nhood[j]] = 0;
            }

            degree1 = min((unsigned)nhood.size(), max_degree);
            out_merge.write((char*)&degree1, sizeof(unsigned));
            out_merge.write((char*)nhood.data(), degree1 * sizeof(unsigned));
            index_size += (uint64_t)((1 + degree1) * sizeof(unsigned));
            nhood.clear();
        }

    }


    cout << endl;
    index_size += 24;

    out_merge.reset();
    out_merge.write((char*)&index_size, sizeof(uint64_t));
    out_merge.write((char*)&max_degree, sizeof(unsigned));
    out_merge.write((char*)&metroids[cluster_num - 1], sizeof(unsigned));
    out_merge.close();
    cout << "Index size:" << index_size << endl;
}
