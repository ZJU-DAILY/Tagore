#include <common.cuh>

using namespace std;

// From the source code of NSG: load data in fvecs format
void load_data(char* filename, float*& data, unsigned& num,unsigned& dim){// load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if(!in.is_open()){std::cout<<"open file error"<<std::endl;exit(-1);}
    in.read((char*)&dim,4);
    in.seekg(0,std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim+1) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0,std::ios::beg);
    for(size_t i = 0; i < num; i++){
        in.seekg(4,std::ios::cur);
        in.read((char*)(data+i*dim),dim*4);
    }
    in.close();
}

// From the source code of NSG: insert neighbors
static inline int InsertIntoPool (Neighbor *addr, unsigned k, Neighbor nn) {
    // find the location to insert
    int left=0,right=k-1;
    if(addr[left].distance>nn.distance){
      memmove((char *)&addr[left+1], &addr[left],k * sizeof(Neighbor));
      addr[left] = nn;
      return left;
    }
    if(addr[right].distance<nn.distance){
      addr[k] = nn;
      return k;
    }
    while(left<right-1){
      int mid=(left+right)/2;
      if(addr[mid].distance>nn.distance)right=mid;
      else left=mid;
    }
    //check equal ID
  
    while (left > 0){
      if (addr[left].distance < nn.distance) break;
      if (addr[left].id == nn.id) return k + 1;
      left--;
    }
    if(addr[left].id == nn.id||addr[right].id==nn.id)return k+1;
    memmove((char *)&addr[right+1], &addr[right],(k-right) * sizeof(Neighbor));
    addr[right]=nn;
    return right;
}

// From the source code of NSG: compute the distance between two vectors
float compare(float* a, float* b, unsigned dim){
    float res = 0.0;
    for(unsigned i = 0; i < dim; i++){
        res += ((a[i] - b[i]) * (a[i] - b[i]));
    }
    return sqrt(res);
}

// From the source code of NSG: perform k-NN search
void get_neighbors(float *query, unsigned* final_graph_, float* data_, unsigned ep_, unsigned nd_, unsigned L, unsigned DIM, unsigned K,
    boost::dynamic_bitset<> &flags,
    std::vector<Neighbor> &retset,
    std::vector<Neighbor> &fullset) {
    retset.resize(L + 1);
    std::vector<unsigned> init_ids(L);

    L = 0;
    for (unsigned i = 0; i < init_ids.size() && i < K; i++) {
        init_ids[i] = final_graph_[ep_ * K + i];
        flags[init_ids[i]] = true;
        L++;
    }
    while (L < init_ids.size()) {
        unsigned id = rand() % nd_;
        if (flags[id]) continue;
        init_ids[L] = id;
        L++;
        flags[id] = true;
    }

    L = 0;
    for (unsigned i = 0; i < init_ids.size(); i++) {
        unsigned id = init_ids[i];
        if (id >= nd_) continue;
        // std::cout<<id<<std::endl;
        float dist = compare(data_ + DIM * id, query, DIM);
        retset[i] = Neighbor(id, dist, true);
        fullset.push_back(retset[i]);
        // flags[id] = 1;
        L++;
    }

    std::sort(retset.begin(), retset.begin() + L);
    int k = 0;
    while (k < (int)L) {
        int nk = L;

        if (retset[k].flag) {
            retset[k].flag = false;
            unsigned n = retset[k].id;

            for (unsigned m = 0; m < K; ++m) {
                unsigned id = final_graph_[n * K + m];
                if (flags[id]) continue;
                flags[id] = 1;

                float dist = compare(query, data_ + DIM * id, DIM);
                Neighbor nn(id, dist, true);
                fullset.push_back(nn);
                if (dist >= retset[L - 1].distance) continue;
                int r = InsertIntoPool(retset.data(), L, nn);

                if (L + 1 < retset.size()) ++L;
                if (r < nk) nk = r;
            }
        }
        if (nk <= k)
        k = nk;
        else
        ++k;
    }
}

// From the source code of NSG: calculate the entry point
void cal_ep(unsigned points, float* data, unsigned& ep_, unsigned* graph, unsigned L, unsigned DIM, unsigned K) {
    float *center = new float[DIM];
    for (unsigned j = 0; j < DIM; j++) center[j] = 0;
    for (unsigned i = 0; i < points; i++) {
    for (unsigned j = 0; j < DIM; j++) {
        center[j] += data[i * DIM + j];
    }
    }
    for (unsigned j = 0; j < DIM; j++) {
    center[j] /= points;
    }
    std::vector<Neighbor> tmp, pool;
    boost::dynamic_bitset<> flags{points, 0};
    ep_ = rand() % points;  // random initialize navigating point
    get_neighbors(center, graph, data, ep_, points, L, DIM, K, flags, tmp, pool);
    ep_ = tmp[0].id;
    delete center;
    cout << "ep: " << ep_ << endl;
}

// store NSG index into files
void StoreNSG(const char *filename, unsigned* graph, unsigned width, unsigned ep_, unsigned points, unsigned K) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    
    out.write((char *)&width, sizeof(unsigned));
    out.write((char *)&ep_, sizeof(unsigned));
    for (unsigned i = 0; i < points; i++) {
      unsigned GK = graph[i * K];
      out.write((char *)&GK, sizeof(unsigned));
      out.write((char *)&graph[i * K + 1], GK * sizeof(unsigned));
    }
    out.close();
}

// store Vamana index into files
void StoreVamana(const char *filename, unsigned* graph, unsigned ep_, unsigned points, size_t num_frozen_points, unsigned K){

    std::ofstream out(filename, std::ios::binary | std::ios::out);

    size_t file_offset = 0;
    out.seekp(file_offset, out.beg);
    size_t index_size = 24;
    uint32_t max_degree = 0;
    out.write((char *)&index_size, sizeof(uint64_t));
    out.write((char *)&max_degree, sizeof(uint32_t));
    uint32_t ep_u32 = ep_;
    out.write((char *)&ep_u32, sizeof(uint32_t));
    out.write((char *)&num_frozen_points, sizeof(size_t));

    unsigned num_points = points + num_frozen_points;
    for (uint32_t i = 0; i < num_points; i++)
    {
        uint32_t GK = (uint32_t)graph[i * K];
        out.write((char *)&GK, sizeof(uint32_t));
        out.write((char *)&graph[i * K + 1], GK * sizeof(uint32_t));
        max_degree = GK > max_degree ? (uint32_t)GK: max_degree;
        index_size += (size_t)(sizeof(uint32_t) * (GK + 1));
    }
    out.seekp(file_offset, out.beg);
    out.write((char *)&index_size, sizeof(uint64_t));
    out.write((char *)&max_degree, sizeof(uint32_t));
    out.close();
}

// store NSSG index into files
void StoreNSSG(const char *filename, unsigned* graph, unsigned width, unsigned points, unsigned K) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    
    std::vector<unsigned> ids(points);
    for(unsigned i=0; i<points; i++){
      ids[i]=i;
    }
    std::random_shuffle(ids.begin(), ids.end());
  
    unsigned n_ep = 10;
    out.write((char *)&width, sizeof(unsigned));
    out.write((char *)&n_ep, sizeof(unsigned));
    out.write((char *)ids.data(), n_ep*sizeof(unsigned));

    for (unsigned i = 0; i < points; i++) {
      unsigned GK = graph[i * K];
      out.write((char *)&GK, sizeof(unsigned));
      out.write((char *)&graph[i * K + 1], GK * sizeof(unsigned));
    }
    out.close();
    cout << endl;
}

// store DPG index into files
void StoreDPG(const char *filename, unsigned* graph, unsigned N, unsigned degree, unsigned K){
    ofstream os(filename, ios::binary);
    char const *KGRAPH_MAGIC = "KNNGRAPH";
    unsigned KGRAPH_MAGIC_SIZE = 8;
    uint32_t VERSION_MAJOR = 2;
    uint32_t VERSION_MINOR = 0;
    os.write(KGRAPH_MAGIC, KGRAPH_MAGIC_SIZE);
    os.write(reinterpret_cast<char const *>(&VERSION_MAJOR), sizeof(VERSION_MAJOR));
    os.write(reinterpret_cast<char const *>(&VERSION_MINOR), sizeof(VERSION_MINOR));
    os.write(reinterpret_cast<char const *>(&N), sizeof(N));
    for (unsigned i = 0; i < N; ++i) {
        uint32_t GK = (uint32_t)graph[i * K];
        
        os.write(reinterpret_cast<char const *>(&GK), sizeof(GK)); 
        os.write(reinterpret_cast<char const *>(&GK), sizeof(GK));
        os.write(reinterpret_cast<char const *>(&graph[i * K + 1]), sizeof(unsigned)*GK);
    }
    os.close();
}

// store CAGRA index into files
void StoreCAGRA(const char* filename, unsigned* graph, unsigned points_num, unsigned degree, unsigned K){
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    for (unsigned i = 0; i < points_num; i++) {
        unsigned GK = (unsigned) degree;
        out.write((char*)&GK, sizeof(unsigned));
        out.write((char*)&graph[i * degree], GK * sizeof(unsigned));
    }
    out.close();
}
