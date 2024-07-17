hydra is a gloo-based communication library system designed to provide multi-rail networks support for allreduce. In addition to supporting TCP networks, it integrates TH Express-2's GLEX network and Mellanox's SHARP network.

Currently, the code uploaded only supports multi-TCP networks architecture, In the future, we will upload the encrypted SHARP and GLEX related code, as well as support for other collective communications on multi-rail networks.

# Compiling hydra Separately for Initial Installation

Follow these steps to compile hydra separately for the first installation:

1. Enter the hydra main directory and execute:

```bash
mkdir build
cd build

2.
cmake ../ [-DUSE_IBVERBS=1] [-DUSE_UCX=1] [-DBUILD_BENCHMARK=1] [-DUSE_REDIS=1]

For example: 
cmake ../ -DUSE_GLEX=1 -DCMAKE_INSTALL_PREFIX=/home/mpi_share/env/hydra_etc/installed/hydra_glex -DBUILD_BENCHMARK=1 -DUSE_REDIS=1 -DHIREDIS_ROOT_DIR=/home/mpi_share/env/hydra_etc/installed/hiredis-1.0.0 -DUSE_SHARP=1 -DUSE_GLEX_RDMA_T=1

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

-s: Number of nodes participating in the operation
-r: the rank value of the current node
-h: node address (host name) where redis is running
-p: The above redis listening port, the default is 6379
-x: The prefix of this operation is mainly used to distinguish different operations. It needs to be changed after each execution, as long as it is different.
-t: Transport layer, such as tcp, glex_rdma_t, etc.
-iteration-count : Number of iterations per data size
bew_allreduce_a: the operation name of the evaluation

5. Examples of experimental results
TCP-4node

Device:      tcp, pci=0000:86:00.0, iface=eth1, addr=[10.0.1.20]
Algorithm:   new_allreduce_ring
Options:     processes=4, inputs=1, threads=1

   elements   min (us)   p50 (us)   p99 (us)  p995 (us)   max (us)   avg (us)   avg (GB/s)    samples
          4        400        684       1093       1686       6256        724        0.000       1000
          8        562        950       1411       2401       7880        965        0.000       1000
         16        525        977       1385       1491       8001        975        0.000       1000
         32        622       1059       1441       1482       4453       1052        0.000       1000
         64        567        953       1416       1473      11775        975        0.000       1000
        128        554       1025       1434       1487      49118       1068        0.000       1000
        256        543        884       1373       1391       1666        915        0.001       1000
        512        520       1009       1431       1453       3267        986        0.002       1000
       1024        549       1024       1493       1625      29379       1066        0.004       1000
       2048        496        852       1358       1393       1465        875        0.009       1000
       4096        613       1263       1620       1900       3694       1224        0.012       1000
       8192        605       1226       1528       1587       4058       1208        0.025       1000
      16384        628       1339       1575       1601       1631       1300        0.047       1000
      32768        590       1226       1608       1700       4985       1197        0.102       1000
      65536       1030       1482       1708       1728       1751       1476        0.165       1000
     131072       1112       1790       2096       2168       2927       1761        0.277       1000
     262144       1496       2007       2848       3145       4105       1993        0.490       1000
     524288       2375       2762       3979       4302       6427       2799        0.698       1000
    1048576       4040       4698       6128       7234      12366       4748        0.823       1000
    2097152       8015       9295      11918      13394      19820       9367        0.834       1000
    4194304      16445      18479      22876      30014      61030      18709        0.835       1000
    8388608      34462      37098      47092      54550      86661      37636        0.830       1000
   16777216      72467      77420      94956      96370     135818      78735        0.794       1000
   33554432     150712     164728     190749     197162     288293     166465        0.751       1000
   67108864     300117     325352     359522     370981     543469     325034        0.769       1000





TCP-TCP-4node

Device:      tcp, pci=0000:d8:00.0, iface=eth2, iface2=eth1, addr=[10.0.0.20]
Algorithm:   bew_allreduce_a
Options:     processes=4, inputs=1, threads=1

   elements   min (us)   p50 (us)   p99 (us)  p995 (us)   max (us)   avg (us)   avg (GB/s)    samples
          4        400        651       1040       1083      29890        707        0.000       1000
          8        515        884       1337       1378       3445        886        0.000       1000
         16        510        893       1258       1307       4246        896        0.000       1000
         32        544       1010       1445       1491      17254       1018        0.000       1000
         64        540       1008       1372       1440       1537        979        0.000       1000
        128        496        778       1200       1254       1390        805        0.001       1000
        256        490        832       1372       1428      39073        913        0.001       1000
        512        526        889       1301       1350       1405        899        0.002       1000
       1024        498        991       1447       1459       2908        990        0.004       1000
       2048        507        909       1335       1361       4880        906        0.008       1000
       4096        542       1195       1526       1561       2077       1146        0.013       1000
       8192        537       1076       1497       1556       5345       1055        0.029       1000
      16384        660       1266       1547       1563       1671       1246        0.049       1000
      32768       1119       1392       1635       1669       2330       1394        0.088       1000
      65536       1213       1464       1695       1740       1876       1466        0.167       1000
     131072        813       1441       2126       5719       6554       1459        0.335       1000
     262144       1419       1940       2443       2521       2655       1955        0.499       1000
     524288       1616       2050       2643       2887      41714       2099        0.930       1000
    1048576       2413       2768       3543       3756       7751       2819        1.386       1000
    2097152       4359       5265       6754       7313      12770       5352        1.459       1000
    4194304       9257      10838      12703      12925      16019      10893        1.434       1000
    8388608      16418      21644      26312      26518      27393      22070        1.416       1000
   16777216      42405      46178      55710      56700      62715      46941        1.331       1000
   33554432      70419      88818     108062     110776     115064      87556        1.428       1000
   67108864     146684     180203     215952     218220     223150     180712        1.383       1000





