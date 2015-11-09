#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_tensor"

#include <boost/test/unit_test.hpp>
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

                BOOST_CHECK_EQUAL(tensor.dims(), dims);
                BOOST_CHECK_EQUAL(tensor.rows(), rows);
                BOOST_CHECK_EQUAL(tensor.cols(), cols);
                BOOST_CHECK_EQUAL(tensor.size(), dims * rows * cols);
                BOOST_CHECK_EQUAL(tensor.planeSize(), rows * cols);

                BOOST_CHECK_EQUAL(tensor.vector().size(), tensor.size());
                BOOST_CHECK_EQUAL(tensor.vector(dims / 2).size(), tensor.planeSize());

                BOOST_CHECK_EQUAL(tensor.matrix(dims - 1).rows(), tensor.rows());
                BOOST_CHECK_EQUAL(tensor.matrix(dims - 1).cols(), tensor.cols());

                tensor.setConstant(constant);

                BOOST_CHECK_EQUAL(tensor.vector().minCoeff(), constant);
                BOOST_CHECK_EQUAL(tensor.vector().maxCoeff(), constant);
        }
}

BOOST_AUTO_TEST_CASE(test_tensor)
{
        const int dims = 4;
        const int rows = 7;
        const int cols = 3;

        check_tensor(dims, rows, cols, 0);
        check_tensor(dims, rows, cols, 1);

        check_tensor(4 * dims, rows, cols, 3.6f);
        check_tensor(dims, 3 * rows, 7 * cols, -2.3);
}

