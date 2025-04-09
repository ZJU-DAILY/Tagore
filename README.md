Tagore: Scalable Graph Indexing using GPUs for Approximate Nearest Neighbor Search
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
  * `data_path` - the path of dataset
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
