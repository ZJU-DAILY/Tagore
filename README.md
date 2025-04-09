<img src="https://www.google.com/url?sa=i&url=https%3A%2F%2Fdics.co%2Fcurrent-affairs%2Fgurudev-rabindranath-tagore&psig=AOvVaw1LR1z5PiWsGa0s5m9JHSRD&ust=1744255250628000&source=images&cd=vfe&opi=89978449&ved=0CBAQjRxqFwoTCNCvnr7_yYwDFQAAAAAdAAAAABAE" width="200" height="100">Tagore: Scalable Graph Indexing using GPUs for Approximate Nearest Neighbor Search
===============================================================================
Tagore is a library that accelerates scalable graph-based index construction for approximate nearest neighbot search using GPUs. It provides user-friendly Python APIs, enabling seamless integration of GPU acceleration into index construction workflows. 
Tagore focuses on the acceleration of refinement-based graph indexes such as NSG, NSSG, Vamana, CAGRA, etc. An efficient kNN graph construction algorithm GNN-Descent, a unified pruning acceleration framework, and an asynchronous GPU-CPU-disk pipeline tailored for billion-scale indexing are included in Tagore. 

Project Building
-------------------------------------------------------------------------------
From the project root directory, do the following:

```
$ mkdir build; cd build
$ cmake ..
$ make
```
After that, a Python package named Tagore will be compiled and stored in the build directory. 

Python API
-------------------------------------------------------------------------------
* `Tagore.GNN_descent(k, num, dim, iteration, data_path)` - construct a kNN graph (initialization phase of refinement-based methods)
  * `k` - degree of the kNN graph
  * `num` - the number of vectors in the used dataset
  * `dim` - the dimension of vectors
  * `iteration` - the number of iterations constructing a kNN graph
  * `data_path` - the path of the dataset
* `Tagore.Pruning(k, num, dim, final_degree, m=64, ptrs, index_type, threshold=1.0, index_path='index.data')` - pruning the initialized kNN graph with a specific method
  * `k` - degree of the kNN graph
  * `num` - the number of vectors in the used dataset
  * `dim` - the dimension of vectors
  * `final_degree` - maximal degree or degree for different index types
  * `m` - the number of neighbor candidates for each node during pruning
  * `ptrs` - pointers of the kNN graph constructed by GNN_descent
  * `index_type` - constructed index type, including 'NSG', 'NSSG', 'Vamana', 'CAGRA', and 'DPG'
  * `threshold` - parameter during pruning, for example $\alpha$ in 'Vamana' and 'NSSG'
  * `index_path` - path to store the index
* `Tagore.Pruning_with_mode(k, num, dim, final_degree, m, ptrs, mode, metric, threshold)` - pruning the initialized kNN graph with the CFS(Collect-Filter-Store) framework introduced in our paper
  * `k` - degree of the kNN graph
  * `num` - the number of vectors in the used dataset
  * `dim` - the dimension of vectors
  * `final_degree` - maximal degree or degree for different index types
  * `m` - the number of neighbor candidates for each node during pruning
  * `ptrs` - pointers of the kNN graph constructed by GNN_descent
  * `mode` - candidates collecting method, including '1-hop', '2-hop', and 'path'
  * `metric` - candidates filtering metric, including 'dist', 'angle', and 'rank'
  * `threshold` - parameter during pruning
* `Tagore.largeIndex(devicelist, dim, k, cluster_num, max_points, data_path_prefix, local_index_path_prefix, index_store_path, max_degree, buffer_size, num, centers_path, threshold, final_degree, m)` - billion-scale index construction method using GPU-accelerated Vamana
  * `devicelist` - ids of GPUs used for indexing
  * `dim` - the dimension of vectors
  * `k` - degree of the kNN graph
  * `cluster_num` - the number of clusters
  * `max_points` - the maximal number of vectors within all clusters
  * `data_path_prefix` - vectors in each cluster are required to be stored in separate files
  * `local_index_path_prefix` - local indexes that are evicted from the cache buffer will be stored in separate files
  *  `index_store_path` - the stored path of the final index
  *  `max_degree` - the maximal degree of the local index
  *  `buffer_size` - the number of local indexes stored in the buffer
  *  `num` -  the number of vectors in the used dataset
  *  `centers_path` - the file path of cluster centers
  *  `threshold` - the value of $\alpha$ in Vamana
  *  `final_degree` - the maximal degree of the final index
  *  `m` - the number of neighbor candidates for each node

 Python Example
 -------------------------------------------------------------------------------
 * Example 1: constructing a million-scale index using NSG 
 ```python
import Tagore

k = 96
vector_num = 1000000
dim = 128
final_degree = 64
m = 64

# Initialization: constructing a kNN graph
gpuPtrs = Tagore.GNN_descent(k, vector_num, dim, 10, 'sift_base.fvecs')

# Pruning: refine the kNN graph using NSG
Tagore.Pruning(k, vector_num, dim, final_degree, m, gpuPtrs, 'NSG')
```

* Example 2: constructing a billion-scale index using Vamana
```python
import Tagore

devicelist = [0, 1, 2, 3] # using 4 GPUs
dim = 96
k = 64
cluster_num = 400
max_points = 5000000
data_path_prefix = '../data/cluster'
local_index_path_prefix = '../index/local_graph'
index_store_path = '../index/final_index.vamana.gpu'
max_degree = 22
buffer_size = 30
vector_num = 1000000000
center_path = '../data/centers.bin'
threshold = 1.2
final_degree = 32
m = 64

Tagore.largeIndex(devicelist, dim, k, cluster_num, max_points, data_path_prefix, local_index_path_prefix, index_store_path, max_degree, buffer_size, vector_num, center_path, threshold, final_degree, m)
