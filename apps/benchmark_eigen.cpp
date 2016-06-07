#include "tensor.h"
#include "text/table.h"
#include "text/cmdline.h"
#include "math/random.hpp"
#include "tensor/numeric.hpp"
#include "cortex/measure.hpp"
#include "text/table_row_mark.h"
#include <iostream>

namespace
{
        using namespace nano;

        nano::random_t<scalar_t> rng(scalar_t(-1e-3), scalar_t(+1e-3));
        const std::size_t trials = 16;

        tensor_size_t measure_dot(const tensor_size_t dims)
        {
                vector_t x(dims);
                vector_t y(dims);
                tensor::set_random(rng, x, y);

                scalar_t z = 0;
                const auto duration = nano::measure_robustly_nsec([&] ()
                {
                        z += x.dot(y);
                }, trials);
                NANO_UNUSED1(z);

                return nano::mflops(2 * dims, duration);
        }

        tensor_size_t measure_sumv1(const tensor_size_t dims)
        {
                vector_t x(dims);
                vector_t z(dims);
                tensor::set_random(rng, x);

                z.setZero();
                const auto duration = nano::measure_robustly_nsec([&] ()
                {
                        z += x * scalar_t(0.5);
                }, trials);
                NANO_UNUSED1(z);

                return nano::mflops(2 * dims, duration);
        }

        tensor_size_t measure_sumv2(const tensor_size_t dims)
        {
                vector_t x(dims);
                vector_t y(dims);
                vector_t z(dims);
                tensor::set_random(rng, x, y);

                z.setZero();
                const auto duration = nano::measure_robustly_nsec([&] ()
                {
                        z += x * scalar_t(0.5) + y * scalar_t(0.3);
                }, trials);
                NANO_UNUSED1(z);

                return nano::mflops(4 * dims, duration);
        }

        tensor_size_t measure_mulv(const tensor_size_t dims)
        {
                matrix_t x(dims, dims);
                vector_t y(dims);
                vector_t z(dims);
                tensor::set_random(rng, x, y);

                z.setZero();
                const auto duration = nano::measure_robustly_nsec([&] ()
                {
                        z += x * y;
                }, trials);
                NANO_UNUSED1(z);

                return nano::mflops(dims * dims + dims, duration);
        }

        tensor_size_t measure_mulm(const tensor_size_t dims)
        {
                matrix_t x(dims, dims);
                matrix_t y(dims, dims);
                matrix_t z(dims, dims);
                tensor::set_random(rng, x, y);

                z.setZero();
                const auto duration = nano::measure_robustly_nsec([&] ()
                {
                        z += x * y;
                }, trials);
                NANO_UNUSED1(z);

                return nano::mflops(dims * dims * dims + dims * dims, duration);
        }

        tensor_size_t measure_outv(const tensor_size_t dims)
        {
                vector_t x(dims);
                vector_t y(dims);
                matrix_t z(dims, dims);
                tensor::set_random(rng, x, y);

                z.setZero();
                const auto duration = nano::measure_robustly_nsec([&] ()
                {
                        z += x * y.transpose();
                }, trials);
                NANO_UNUSED1(z);

                return nano::mflops(2 * dims * dims, duration);
        }
}

int main(int, const char* [])
{
        using namespace nano;

        const auto min_dims = tensor_size_t(8);
        const auto max_dims = tensor_size_t(1024);
        const auto foreach_dims = [&] (const auto& op)
        {
                for (tensor_size_t dims = min_dims; dims <= max_dims; dims *= 2)
                {
                        op(dims);
                }
        };
        const auto fillrow = [&] (auto&& row, const auto& op)
        {
                foreach_dims([&] (const auto dims) { row << op(dims); });
        };

        table_t table("operation\\dimensions [MFLOPS]");
        foreach_dims([&] (const auto dims) { table.header() << to_string(dims); });

        fillrow(table.append("z += x.dot(y)"), measure_dot);
        fillrow(table.append("z += x * 0.5"), measure_sumv1);
        fillrow(table.append("z += x * 0.5 + y * 0.3"), measure_sumv2);
        fillrow(table.append("z += X * y"), measure_mulv);
        fillrow(table.append("Z += X * Y"), measure_mulm);
        fillrow(table.append("Z += x * y^t"), measure_outv);

        table.mark(nano::make_table_mark_maximum_percentage_cols<size_t>(10));
        table.print(std::cout);

        return EXIT_SUCCESS;
}

