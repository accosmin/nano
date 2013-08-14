# no hidden layer
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -s 1000
echo

# one hidden layer, no pooling
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;anorm" -s 10000
echo
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;snorm" -s 10000
echo
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;tanh" -s 10000
echo
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;unit" -s 10000
echo

# one hidden layer, pooling
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;anorm;max-pool" -s 10000
echo
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;snorm;max-pool" -s 10000
echo
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;tanh;max-pool" -s 10000
echo
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;unit;max-pool" -s 10000
echo

# two hidden layers, no pooling
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;anorm;conv:convs=16,crows=8,ccols=8;anorm" -s 10000
echo
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;snorm;conv:convs=16,crows=8,ccols=8;snorm" -s 10000
echo
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;tanh;conv:convs=16,crows=8,ccols=8;tanh" -s 10000
echo
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;unit;conv:convs=16,crows=8,ccols=8;unit" -s 10000
echo

# two hidden layers, pooling
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;anorm;max-pool;conv:convs=16,crows=8,ccols=8;anorm;max-pool" -s 10000
echo
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;snorm;max-pool;conv:convs=16,crows=8,ccols=8;snorm;max-pool" -s 10000
echo
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;tanh;max-pool;conv:convs=16,crows=8,ccols=8;tanh;max-pool" -s 10000
echo
./build/ncv_test_network_forward -i rgba -r 32 -c 32 -o 10 -n "conv:convs=16,crows=8,ccols=8;unit;max-pool;conv:convs=16,crows=8,ccols=8;unit;max-pool" -s 10000
echo
