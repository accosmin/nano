#include "unit_test.hpp"
#include "tensor/tensor.hpp"

NANO_BEGIN_MODULE(test_tensor)

NANO_CASE(index1d)
{
        tensor::tensor_index_t<std::int64_t, 1> index(7);

        NANO_CHECK_EQUAL(index.size(), 7);
        NANO_CHECK_EQUAL(index.size<0>(), 7);

        NANO_CHECK_EQUAL(index(0), 0);
        NANO_CHECK_EQUAL(index(1), 1);
        NANO_CHECK_EQUAL(index(6), 6);
}

NANO_CASE(index2d)
{
        tensor::tensor_index_t<std::int64_t, 2> index(7, 5);

        NANO_CHECK_EQUAL(index.size(), 35);
        NANO_CHECK_EQUAL(index.size<0>(), 7);
        NANO_CHECK_EQUAL(index.size<1>(), 5);

        NANO_CHECK_EQUAL(index(0), 0);
        NANO_CHECK_EQUAL(index(13), 13);
        NANO_CHECK_EQUAL(index(34), 34);

        NANO_CHECK_EQUAL(index(0, 1), 1);
        NANO_CHECK_EQUAL(index(0, 4), 4);
        NANO_CHECK_EQUAL(index(1, 0), 5);
        NANO_CHECK_EQUAL(index(3, 2), 17);
        NANO_CHECK_EQUAL(index(6, 4), 34);
}

NANO_CASE(index3d)
{
        tensor::tensor_index_t<std::int64_t, 3> index(3, 7, 5);

        NANO_CHECK_EQUAL(index.size(), 105);
        NANO_CHECK_EQUAL(index.size<0>(), 3);
        NANO_CHECK_EQUAL(index.size<1>(), 7);
        NANO_CHECK_EQUAL(index.size<2>(), 5);

        NANO_CHECK_EQUAL(index(0), 0);
        NANO_CHECK_EQUAL(index(13), 13);
        NANO_CHECK_EQUAL(index(34), 34);

        NANO_CHECK_EQUAL(index(0, 0, 1), 1);
        NANO_CHECK_EQUAL(index(0, 0, 4), 4);
        NANO_CHECK_EQUAL(index(0, 1, 0), 5);
        NANO_CHECK_EQUAL(index(0, 2, 1), 11);
        NANO_CHECK_EQUAL(index(1, 2, 1), 46);
        NANO_CHECK_EQUAL(index(1, 0, 3), 38);
        NANO_CHECK_EQUAL(index(2, 4, 1), 91);
        NANO_CHECK_EQUAL(index(2, 6, 4), 104);
}

NANO_CASE(tensor3d)
{
        using tensor3d_t = tensor::tensor_t<float, 3>;

        const int dims = 7;
        const int rows = 3;
        const int cols = 4;

        tensor3d_t tensor;
        tensor.resize(dims, rows, cols);

        NANO_CHECK_EQUAL(tensor.dims(), dims);
        NANO_CHECK_EQUAL(tensor.rows(), rows);
        NANO_CHECK_EQUAL(tensor.cols(), cols);
        NANO_CHECK_EQUAL(tensor.size(), dims * rows * cols);
        NANO_CHECK_EQUAL(tensor.planeSize(), rows * cols);

        NANO_CHECK_EQUAL(tensor.vector().size(), tensor.size());
        NANO_CHECK_EQUAL(tensor.vector(dims / 2).size(), tensor.planeSize());

        NANO_CHECK_EQUAL(tensor.matrix(dims - 1).rows(), tensor.rows());
        NANO_CHECK_EQUAL(tensor.matrix(dims - 1).cols(), tensor.cols());

        tensor.setConstant(constant);

        NANO_CHECK_EQUAL(tensor.vector().minCoeff(), constant);
        NANO_CHECK_EQUAL(tensor.vector().maxCoeff(), constant);
}

NANO_END_MODULE()

