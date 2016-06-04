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

        tensor_size_t measure_dot(const tensor_size_t dims, const std::size_t trials = 16)
        {
                nano::random_t<scalar_t> rng(scalar_t(-0.1) / dims, scalar_t(+0.1) / dims);

                vector_t x(dims);
                vector_t y(dims);
                tensor::set_random(rng, x, y);

                scalar_t sum = 0;
                const auto duration = nano::measure_robustly_nsec([&] ()
                {
                        sum += x.dot(y);
                }, trials);
                NANO_UNUSED1(sum);

                return duration.count();//nano::mflops(2 * dims, duration);
        }
}

int main(int, const char* [])
{
        using namespace nano;

        table_t table("operation\\dimensions [MFLOPS]");
        table.header() << "10" << "100" << "1000" << "10000";

        table.append("v = x.dot(y)")
                << measure_dot(10)
                << measure_dot(100)
                << measure_dot(1000)
                << measure_dot(10000);

        table.mark(nano::make_table_mark_maximum_percentage_cols<size_t>(10));
        table.print(std::cout);

        return EXIT_SUCCESS;
}

