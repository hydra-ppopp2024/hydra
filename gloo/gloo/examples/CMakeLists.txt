
include_directories(/home/mpi_share/env/yed_gloo/gloo_glex/installed/gloo_glex/include)
include_directories(/home/mpi_share/env/mlnx_sharp/hpcx-v2.7.0-gcc-MLNX_OFED_LINUX-5.1-0.6.6.0-redhat7.8-x86_64/sharp/include)
include_directories(/usr/local/glex/include)
include_directories(/home/mpi_share/env/gloo_etc/installed/hiredis-1.0.0/include/hiredis)
link_directories(/home/mpi_share/env/yed_gloo/gloo_glex/installed/gloo_glex/lib)
link_directories(/home/mpi_share/env/mlnx_sharp/hpcx-v2.7.0-gcc-MLNX_OFED_LINUX-5.1-0.6.6.0-redhat7.8-x86_64/sharp/lib)
link_directories(/usr/local/glex/lib)
link_directories(/home/mpi_share/env/gloo_etc/installed/hiredis-1.0.0/lib)

add_executable(example_all example_allreduce.cc)
target_link_libraries(example_all gloo)
target_link_libraries(example_all pthread)
# add_executable(looks_like_mpi looks_like_mpi.cc)
# target_link_libraries(looks_like_mpi gloo)
# target_link_libraries(looks_like_mpi pthread)
# add_executable(example_sharp_allreduce1 example_sharp_allreduce.cc)
# target_link_libraries(example_sharp_allreduce1 gloo)
# target_link_libraries(example_sharp_allreduce1 pthread)
# target_link_libraries(example_sharp_allreduce1 sharp)
# target_link_libraries(example_sharp_allreduce1 sharp_coll)

#---------------------------------------------------

#--------------------------------------------------
add_executable(example_glex example_glex_redis.cc)
target_link_libraries(example_glex gloo)
target_link_libraries(example_glex glex)
target_link_libraries(example_glex pthread)
target_link_libraries(example_glex hiredis)
#-----------------------

