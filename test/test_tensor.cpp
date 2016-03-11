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

