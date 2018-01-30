#include "utest.h"
#include "tensor/tensor.h"
#include <vector>

using namespace nano;

NANO_BEGIN_MODULE(test_tensor)

NANO_CASE(index1d)
{
        const auto dims = nano::make_dims(7);

        NANO_CHECK_EQUAL(std::get<0>(dims), 7);
        NANO_CHECK_EQUAL(nano::size(dims), 7);

        NANO_CHECK_EQUAL(nano::index(dims, 0), 0);
        NANO_CHECK_EQUAL(nano::index(dims, 1), 1);
        NANO_CHECK_EQUAL(nano::index(dims, 6), 6);

        NANO_CHECK_EQUAL(nano::index0(dims), nano::index(dims, 0));
        NANO_CHECK_EQUAL(nano::index0(dims, 6), nano::index(dims, 6));

        NANO_CHECK_EQUAL(nano::dims0(dims), nano::make_dims(7));
}

NANO_CASE(index2d)
{
        const auto dims = nano::make_dims(7, 5);

        NANO_CHECK_EQUAL(dims, nano::cat_dims(7, nano::make_dims(5)));

        NANO_CHECK_EQUAL(std::get<0>(dims), 7);
        NANO_CHECK_EQUAL(std::get<1>(dims), 5);
        NANO_CHECK_EQUAL(nano::size(dims), 35);

        NANO_CHECK_EQUAL(nano::index(dims, 0, 1), 1);
        NANO_CHECK_EQUAL(nano::index(dims, 0, 4), 4);
        NANO_CHECK_EQUAL(nano::index(dims, 1, 0), 5);
        NANO_CHECK_EQUAL(nano::index(dims, 3, 2), 17);
        NANO_CHECK_EQUAL(nano::index(dims, 6, 4), 34);

        NANO_CHECK_EQUAL(nano::index0(dims), nano::index(dims, 0, 0));
        NANO_CHECK_EQUAL(nano::index0(dims, 3), nano::index(dims, 3, 0));
        NANO_CHECK_EQUAL(nano::index0(dims, 3, 1), nano::index(dims, 3, 1));

        NANO_CHECK_EQUAL(nano::dims0(dims), nano::make_dims(7, 5));
        NANO_CHECK_EQUAL(nano::dims0(dims, 3), nano::make_dims(5));
}

NANO_CASE(index3d)
{
        const auto dims = nano::make_dims(3, 7, 5);

        NANO_CHECK_EQUAL(dims, nano::cat_dims(3, nano::make_dims(7, 5)));

        NANO_CHECK_EQUAL(std::get<0>(dims), 3);
        NANO_CHECK_EQUAL(std::get<1>(dims), 7);
        NANO_CHECK_EQUAL(std::get<2>(dims), 5);
        NANO_CHECK_EQUAL(nano::size(dims), 105);

        NANO_CHECK_EQUAL(nano::index(dims, 0, 0, 1), 1);
        NANO_CHECK_EQUAL(nano::index(dims, 0, 0, 4), 4);
        NANO_CHECK_EQUAL(nano::index(dims, 0, 1, 0), 5);
        NANO_CHECK_EQUAL(nano::index(dims, 0, 2, 1), 11);
        NANO_CHECK_EQUAL(nano::index(dims, 1, 2, 1), 46);
        NANO_CHECK_EQUAL(nano::index(dims, 1, 0, 3), 38);
        NANO_CHECK_EQUAL(nano::index(dims, 2, 4, 1), 91);
        NANO_CHECK_EQUAL(nano::index(dims, 2, 6, 4), 104);

        NANO_CHECK_EQUAL(nano::index0(dims), nano::index(dims, 0, 0, 0));
        NANO_CHECK_EQUAL(nano::index0(dims, 2), nano::index(dims, 2, 0, 0));
        NANO_CHECK_EQUAL(nano::index0(dims, 2, 4), nano::index(dims, 2, 4, 0));
        NANO_CHECK_EQUAL(nano::index0(dims, 2, 4, 3), nano::index(dims, 2, 4, 3));

        NANO_CHECK_EQUAL(nano::dims0(dims), nano::make_dims(3, 7, 5));
        NANO_CHECK_EQUAL(nano::dims0(dims, 2), nano::make_dims(7, 5));
        NANO_CHECK_EQUAL(nano::dims0(dims, 2, 4), nano::make_dims(5));
}

NANO_CASE(isfinite)
{
        using tensor3d_t = nano::tensor_mem_t<float, 3>;

        const auto dims = 7;
        const auto rows = 3;
        const auto cols = 4;

        tensor3d_t tensor;
        tensor.resize(dims, rows, cols);

        tensor.zero();
        NANO_CHECK(isfinite(tensor));

        tensor(1, 2, 3) = NAN;
        NANO_CHECK(!isfinite(tensor));

        tensor.zero();
        tensor(1, 2, 3) = INFINITY;
        NANO_CHECK(!isfinite(tensor));
}

NANO_CASE(tensor3d)
{
        using tensor3d_t = nano::tensor_mem_t<int, 3>;

        const auto dims = 7;
        const auto rows = 3;
        const auto cols = 4;

        tensor3d_t tensor;
        tensor.resize(dims, rows, cols);

        tensor.zero();
        NANO_CHECK_EQUAL(tensor.vector().minCoeff(), 0);
        NANO_CHECK_EQUAL(tensor.vector().maxCoeff(), 0);

        NANO_CHECK_EQUAL(tensor.size<0>(), dims);
        NANO_CHECK_EQUAL(tensor.size<1>(), rows);
        NANO_CHECK_EQUAL(tensor.size<2>(), cols);
        NANO_CHECK_EQUAL(tensor.rows(), rows);
        NANO_CHECK_EQUAL(tensor.cols(), cols);
        NANO_CHECK_EQUAL(tensor.size(), dims * rows * cols);

        NANO_CHECK_EQUAL(tensor.vector().size(), dims * rows * cols);
        NANO_CHECK_EQUAL(tensor.vector(dims / 2).size(), rows * cols);
        NANO_CHECK_EQUAL(tensor.vector(dims / 2, rows / 2).size(), cols);

        NANO_CHECK_EQUAL(tensor.matrix(dims - 1).rows(), tensor.rows());
        NANO_CHECK_EQUAL(tensor.matrix(dims - 1).cols(), tensor.cols());

        tensor(0, 0, 1) = -3;
        tensor(2, 2, 0) = -7;
        NANO_CHECK_EQUAL(tensor(0, 0, 1), -3);
        NANO_CHECK_EQUAL(tensor(2, 2, 0), -7);

        tensor.constant(42);
        NANO_CHECK_EQUAL(tensor.vector().minCoeff(), 42);
        NANO_CHECK_EQUAL(tensor.vector().maxCoeff(), 42);

        tensor.constant(42);
        tensor.vector(3, 0).setConstant(7);
        NANO_CHECK_EQUAL(tensor.vector().minCoeff(), 7);
        NANO_CHECK_EQUAL(tensor.vector().maxCoeff(), 42);
        NANO_CHECK_EQUAL(tensor.vector().sum(), 42 * dims * rows * cols - (42 - 7) * cols);

        tensor.matrix(3).setConstant(13);
        NANO_CHECK_EQUAL(tensor.matrix(3).minCoeff(), 13);
        NANO_CHECK_EQUAL(tensor.matrix(3).maxCoeff(), 13);
}

NANO_CASE(tensor3d_map)
{
        using tensor3d_t = nano::tensor_mem_t<int, 3>;

        const auto dims = 7;
        const auto rows = 3;
        const auto cols = 4;

        tensor3d_t tensor;
        tensor.resize(dims + 1, rows - 3, cols + 2);

        std::vector<int> v;
        v.reserve(dims * rows * cols);
        for (int i = 0; i < dims * rows * cols; ++ i)
        {
                v.push_back(-35 + i);
        }

        const auto tmap = ::nano::map_tensor(v.data(), dims, rows, cols);
        NANO_CHECK_EQUAL(tmap.size<0>(), dims);
        NANO_CHECK_EQUAL(tmap.size<1>(), rows);
        NANO_CHECK_EQUAL(tmap.size<2>(), cols);
        NANO_CHECK_EQUAL(tmap.rows(), rows);
        NANO_CHECK_EQUAL(tmap.cols(), cols);
        NANO_CHECK_EQUAL(tmap.size(), dims * rows * cols);

        for (int d = 0, i = 0; d < dims; ++ d)
        {
                for (int r = 0; r < rows; ++ r)
                {
                        for (int c = 0; c < cols; ++ c, ++ i)
                        {
                                NANO_CHECK_EQUAL(tmap(d, r, c), -35 + i);
                        }
                }
        }

        for (int i = 0; i < tmap.size(); ++ i)
        {
                NANO_CHECK_EQUAL(tmap(i), -35 + i);
        }

        tensor = tmap;
        NANO_CHECK_EQUAL(tensor.size<0>(), dims);
        NANO_CHECK_EQUAL(tensor.size<1>(), rows);
        NANO_CHECK_EQUAL(tensor.size<2>(), cols);
        NANO_CHECK_EQUAL(tensor.rows(), rows);
        NANO_CHECK_EQUAL(tensor.cols(), cols);

        for (int d = 0, i = 0; d < dims; ++ d)
        {
                for (int r = 0; r < rows; ++ r)
                {
                        for (int c = 0; c < cols; ++ c, ++ i)
                        {
                                NANO_CHECK_EQUAL(tensor(d, r, c), -35 + i);
                        }
                }
        }

        for (int i = 0; i < tensor.size(); ++ i)
        {
                NANO_CHECK_EQUAL(tensor(i), -35 + i);
        }
}

NANO_CASE(tensor4d)
{
        using tensor4d_t = nano::tensor_mem_t<int, 4>;

        const auto dim1 = 2;
        const auto dim2 = 7;
        const auto rows = 3;
        const auto cols = 4;

        tensor4d_t tensor;
        tensor.resize(dim1, dim2, rows, cols);

        tensor.zero();
        NANO_CHECK_EQUAL(tensor.vector().minCoeff(), 0);
        NANO_CHECK_EQUAL(tensor.vector().maxCoeff(), 0);

        NANO_CHECK_EQUAL(tensor.size<0>(), dim1);
        NANO_CHECK_EQUAL(tensor.size<1>(), dim2);
        NANO_CHECK_EQUAL(tensor.size<2>(), rows);
        NANO_CHECK_EQUAL(tensor.size<3>(), cols);
        NANO_CHECK_EQUAL(tensor.rows(), rows);
        NANO_CHECK_EQUAL(tensor.cols(), cols);
        NANO_CHECK_EQUAL(tensor.size(), dim1 * dim2 * rows * cols);

        NANO_CHECK_EQUAL(tensor.vector().size(), dim1 * dim2 * rows * cols);
        NANO_CHECK_EQUAL(tensor.vector(dim1 / 2).size(), dim2 * rows * cols);
        NANO_CHECK_EQUAL(tensor.vector(dim1 / 2, dim2 / 2).size(), rows * cols);
        NANO_CHECK_EQUAL(tensor.vector(dim1 / 2, dim2 / 2, rows / 2).size(), cols);

        NANO_CHECK_EQUAL(tensor.matrix(dim1 - 1, dim2 - 1).rows(), tensor.rows());
        NANO_CHECK_EQUAL(tensor.matrix(dim1 - 1, dim2 - 1).cols(), tensor.cols());

        tensor(0, 4, 0, 1) = -3;
        tensor(1, 2, 2, 0) = -7;
        NANO_CHECK_EQUAL(tensor(0, 4, 0, 1), -3);
        NANO_CHECK_EQUAL(tensor(1, 2, 2, 0), -7);

        tensor.constant(42);
        NANO_CHECK_EQUAL(tensor.vector().minCoeff(), 42);
        NANO_CHECK_EQUAL(tensor.vector().maxCoeff(), 42);

        tensor.vector(0, 3).setConstant(7);
        NANO_CHECK_EQUAL(tensor.vector().minCoeff(), 7);
        NANO_CHECK_EQUAL(tensor.vector().maxCoeff(), 42);
        NANO_CHECK_EQUAL(tensor.vector().sum(), 42 * dim1 * dim2 * rows * cols - (42 - 7) * rows * cols);


        tensor.matrix(0, 3).setConstant(13);
        NANO_CHECK_EQUAL(tensor.matrix(0, 3).minCoeff(), 13);
        NANO_CHECK_EQUAL(tensor.matrix(0, 3).maxCoeff(), 13);
}

NANO_CASE(tensor4d_map)
{
        using tensor4d_t = nano::tensor_mem_t<int, 4>;

        const auto dim1 = 3;
        const auto dim2 = 7;
        const auto rows = 3;
        const auto cols = 4;

        tensor4d_t tensor;
        tensor.resize(dim1 + 2, dim2 + 1, rows - 3, cols + 2);

        std::vector<int> v;
        v.reserve(dim1 * dim2 * rows * cols);
        for (int i = 0; i < dim1 * dim2 * rows * cols; ++ i)
        {
                v.push_back(-35 + i);
        }

        const auto tmap = ::nano::map_tensor(v.data(), dim1, dim2, rows, cols);
        NANO_CHECK_EQUAL(tmap.size<0>(), dim1);
        NANO_CHECK_EQUAL(tmap.size<1>(), dim2);
        NANO_CHECK_EQUAL(tmap.size<2>(), rows);
        NANO_CHECK_EQUAL(tmap.size<3>(), cols);
        NANO_CHECK_EQUAL(tmap.rows(), rows);
        NANO_CHECK_EQUAL(tmap.cols(), cols);
        NANO_CHECK_EQUAL(tmap.size(), dim1 * dim2 * rows * cols);

        for (int d1 = 0, i = 0; d1 < dim1; ++ d1)
        {
                for (int d2 = 0; d2 < dim2; ++ d2)
                {
                        for (int r = 0; r < rows; ++ r)
                        {
                                for (int c = 0; c < cols; ++ c, ++ i)
                                {
                                        NANO_CHECK_EQUAL(tmap(d1, d2, r, c), -35 + i);
                                }
                        }
                }
        }

        for (int i = 0; i < tmap.size(); ++ i)
        {
                NANO_CHECK_EQUAL(tmap(i), -35 + i);
        }

        tensor = tmap;
        NANO_CHECK_EQUAL(tensor.size<0>(), dim1);
        NANO_CHECK_EQUAL(tensor.size<1>(), dim2);
        NANO_CHECK_EQUAL(tensor.size<2>(), rows);
        NANO_CHECK_EQUAL(tensor.size<3>(), cols);
        NANO_CHECK_EQUAL(tensor.rows(), rows);
        NANO_CHECK_EQUAL(tensor.cols(), cols);

        for (int d1 = 0, i = 0; d1 < dim1; ++ d1)
        {
                for (int d2 = 0; d2 < dim2; ++ d2)
                {
                        for (int r = 0; r < rows; ++ r)
                        {
                                for (int c = 0; c < cols; ++ c, ++ i)
                                {
                                        NANO_CHECK_EQUAL(tensor(d1, d2, r, c), -35 + i);
                                }
                        }
                }
        }

        for (int i = 0; i < tensor.size(); ++ i)
        {
                NANO_CHECK_EQUAL(tensor(i), -35 + i);
        }
}

NANO_CASE(tensor3d_fill)
{
        using tensor3d_t = nano::tensor_mem_t<double, 3>;

        const auto dims = 7;
        const auto rows = 3;
        const auto cols = 4;

        tensor3d_t tensor;
        tensor.resize(dims, rows, cols);

        tensor.zero();
        NANO_CHECK_EQUAL(tensor.vector().minCoeff(), 0);
        NANO_CHECK_EQUAL(tensor.vector().maxCoeff(), 0);

        tensor.constant(-4);
        NANO_CHECK_EQUAL(tensor.vector().minCoeff(), -4);
        NANO_CHECK_EQUAL(tensor.vector().maxCoeff(), -4);

        tensor.random(-3, +5);
        NANO_CHECK_GREATER(tensor.vector().minCoeff(), -3);
        NANO_CHECK_LESS(tensor.vector().maxCoeff(), +5);

        tensor.random(+5, +11);
        NANO_CHECK_GREATER(tensor.vector().minCoeff(), +5);
        NANO_CHECK_LESS(tensor.vector().maxCoeff(), +11);
}

NANO_CASE(tensor4d_reshape)
{
        using tensor4d_t = nano::tensor_mem_t<int, 4>;

        tensor4d_t tensor(5, 6, 7, 8);

        auto reshape4d = tensor.reshape(5, 3, 28, 4);
        NANO_CHECK_EQUAL(reshape4d.data(), tensor.data());
        NANO_CHECK_EQUAL(reshape4d.size(), tensor.size());
        NANO_CHECK_EQUAL(reshape4d.size<0>(), 5);
        NANO_CHECK_EQUAL(reshape4d.size<1>(), 3);
        NANO_CHECK_EQUAL(reshape4d.size<2>(), 28);
        NANO_CHECK_EQUAL(reshape4d.size<3>(), 4);

        auto reshape3d = tensor.reshape(30, 14, 4);
        NANO_CHECK_EQUAL(reshape3d.data(), tensor.data());
        NANO_CHECK_EQUAL(reshape3d.size(), tensor.size());
        NANO_CHECK_EQUAL(reshape3d.size<0>(), 30);
        NANO_CHECK_EQUAL(reshape3d.size<1>(), 14);
        NANO_CHECK_EQUAL(reshape3d.size<2>(), 4);

        auto reshape2d = tensor.reshape(30, 56);
        NANO_CHECK_EQUAL(reshape2d.data(), tensor.data());
        NANO_CHECK_EQUAL(reshape2d.size(), tensor.size());
        NANO_CHECK_EQUAL(reshape2d.size<0>(), 30);
        NANO_CHECK_EQUAL(reshape2d.size<1>(), 56);

        auto reshape1d = tensor.reshape(1680);
        NANO_CHECK_EQUAL(reshape1d.data(), tensor.data());
        NANO_CHECK_EQUAL(reshape1d.size(), tensor.size());
        NANO_CHECK_EQUAL(reshape1d.size<0>(), 1680);
}

NANO_CASE(tensor4d_subtensor)
{
        using tensor4d_t = nano::tensor_mem_t<int, 4>;

        const auto dim1 = 2;
        const auto dim2 = 7;
        const auto rows = 3;
        const auto cols = 4;

        tensor4d_t tensor;
        tensor.resize(dim1, dim2, rows, cols);

        tensor.constant(42);
        NANO_CHECK_EQUAL(tensor.vector().minCoeff(), 42);
        NANO_CHECK_EQUAL(tensor.vector().maxCoeff(), 42);

        tensor.constant(42);
        tensor.tensor(1, 2).setConstant(7);
        NANO_CHECK_EQUAL(tensor.tensor(1, 2).dims(), nano::make_dims(rows, cols));
        NANO_CHECK_EQUAL(tensor.tensor(1, 2).array().minCoeff(), 7);
        NANO_CHECK_EQUAL(tensor.tensor(1, 2).array().maxCoeff(), 7);
        NANO_CHECK_EQUAL(tensor.tensor(1, 2).array().sum(), 7 * rows * cols);
        NANO_CHECK_EQUAL(tensor.vector().sum(), 42 * dim1 * dim2 * rows * cols - (42 - 7) * rows * cols);

        tensor.constant(42);
        tensor.tensor(1).setConstant(7);
        NANO_CHECK_EQUAL(tensor.tensor(1).dims(), nano::make_dims(dim2, rows, cols));
        NANO_CHECK_EQUAL(tensor.tensor(1).array().minCoeff(), 7);
        NANO_CHECK_EQUAL(tensor.tensor(1).array().maxCoeff(), 7);
        NANO_CHECK_EQUAL(tensor.tensor(1).array().sum(), 7 * dim2 * rows * cols);
        NANO_CHECK_EQUAL(tensor.vector().sum(), 42 * dim1 * dim2 * rows * cols - (42 - 7) * dim2 * rows * cols);
}

NANO_END_MODULE()
