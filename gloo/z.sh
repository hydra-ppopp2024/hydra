#!/bin/bash
rm -rf build
mkdir build
cd build
cmake ../ -DUSE_GLEX=1 -DCMAKE_INSTALL_PREFIX=/home/mpi_share/env/yed_gloo/gloo_glex/installed/gloo_glex -DBUILD_BENCHMARK=1 -DUSE_REDIS=1 -DHIREDIS_ROOT_DIR=/home/mpi_share/env/gloo_etc/installed/hiredis-1.0.0 -DUSE_SHARP=1 -DUSE_GLEX_RDMA_T=1 -DUSE_GLEX_RDMA_S=1
make -j8
make install
