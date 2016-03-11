#include "unit_test.hpp"
#include "tensor/tensor.hpp"

namespace
{
        template
        <
                typename tscalar
        >
        void check_tensor(int dims, int rows, int cols, tscalar constant)
        {
                tensor::tensor_t<tscalar> tensor;

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
}

NANO_BEGIN_MODULE(test_tensor)

NANO_CASE(index1d)
{
        tensor::tensor_index_t<int, 1> index(7);

        NANO_CHECK_EQUAL(index.size(), 7);
        NANO_CHECK_EQUAL(index.size(0), 7);

        NANO_CHECK_EQUAL(index(0), 0);
        NANO_CHECK_EQUAL(index(1), 1);
        NANO_CHECK_EQUAL(index(6), 6);
}

NANO_CASE(index2d)
{
        tensor::tensor_index_t<int, 2> index(7, 5);

        NANO_CHECK_EQUAL(index.size(), 35);
        NANO_CHECK_EQUAL(index.size(0), 7);
        NANO_CHECK_EQUAL(index.size(1), 5);

        NANO_CHECK_EQUAL(index(0), 0);
        NANO_CHECK_EQUAL(index(13), 13);
        NANO_CHECK_EQUAL(index(34), 34);

        NANO_CHECK_EQUAL(index(0, 1), 1);
        NANO_CHECK_EQUAL(index(0, 4), 4);
        NANO_CHECK_EQUAL(index(1, 0), 5);
        NANO_CHECK_EQUAL(index(3, 2), 17);
        NANO_CHECK_EQUAL(index(6, 4), 34);
}

NANO_CASE(construction)
{
        const int dims = 4;
        const int rows = 7;
        const int cols = 3;

        check_tensor(dims, rows, cols, 0);
        check_tensor(dims, rows, cols, 1);

        check_tensor(4 * dims, rows, cols, 3.6f);
        check_tensor(dims, 3 * rows, 7 * cols, -2.3);
}

NANO_END_MODULE()

