Nezha is a gloo-based communication library system designed to provide multi-rail networks support for allreduce. In addition to supporting TCP networks, it integrates TH Express-2's GLEX network and Mellanox's SHARP network.

Currently, the code uploaded only supports multi-TCP networks architecture, In the future, we will upload the encrypted SHARP and GLEX related code, as well as support for other collective communications on multi-rail networks.

# Compiling Nezha Separately for Initial Installation

Follow these steps to compile Nezha separately for the first installation:

1. Enter the Nezha main directory and execute:

```bash
mkdir build
cd build

2.
cmake ../ [-DUSE_IBVERBS=1] [-DUSE_UCX=1] [-DBUILD_BENCHMARK=1] [-DUSE_REDIS=1]

For example: 
cmake ../ -DUSE_GLEX=1 -DCMAKE_INSTALL_PREFIX=/home/mpi_share/env/nezha_etc/installed/nezha_glex -DBUILD_BENCHMARK=1 -DUSE_REDIS=1 -DHIREDIS_ROOT_DIR=/home/mpi_share/env/nezha_etc/installed/hiredis-1.0.0 -DUSE_SHARP=1 -DUSE_GLEX_RDMA_T=1

Explanation of options:
- DUSE_GLEX and DUSE_GLEX_RDMA_T add underlying communication libraries (GLEX for MP messages, GLEX_RDMA_T for RDMA).
- DUSE_SHARP enables SHArP support for in-network computing.
- DBUILD_BENCHMARK and DUSE_REDIS enable the built-in benchmark and Redis support.

You can modify the CMakeLists.txt file to add new compilation options.

3. Compile and install:

```bash
make -j8
make install

4. Experiment:
Write test programs in the ./gloo/examples/ directory.
Use the provided benchmark by installing Redis for address transmission.
To start Redis, go to the Redis directory and execute ./redis-server.

Run the benchmark program from the ./build/gloo/benchmark/ directory with parameters like -s, -r, -h, -p, -x, -t, and --iteration-count.

Example: 
ALLREDUCE_ALLREDUCE_=1  ./benchmark -s 2 -r 0 -h cn0 -p 6379 -x 210 -t tcp --tcp-device eth2    -u tcp --tcp-device2 eth1   --iteration-count 1000 bew_allreduce_a 
ALLREDUCE_ALLREDUCE_=1  ./benchmark -s 2 -r 1 -h cn0 -p 6379 -x 210 -t tcp --tcp-device eth2    -u tcp --tcp-device2 eth1    --iteration-count 1000 bew_allreduce_a


