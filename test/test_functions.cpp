#include "utest.h"
#include "math/random.h"
#include "math/epsilon.h"
#include "functions/test.h"
#include "tensor/numeric.h"

using namespace nano;

static void test_function(const function_t& function)
{
        std::cout << function.name() << std::endl;

        const auto trials = size_t(1000);

        const auto dims = function.size();
        NANO_CHECK_GREATER(dims, 0);
        NANO_CHECK_GREATER_EQUAL(dims, function.min_size());
        NANO_CHECK_GREATER_EQUAL(function.max_size(), dims);

        auto rgen = make_rng(scalar_t(-10.0), scalar_t(+10.0));

        for (size_t t = 0; t < trials; ++ t)
        {
                vector_t x0(dims), x1(dims);
                nano::set_random(rgen, x0, x1);

                if (    function.is_valid(x0) && std::isfinite(function.eval(x0)) &&
                        function.is_valid(x1) && std::isfinite(function.eval(x1)))
                {
                        NANO_CHECK_LESS(function.grad_accuracy(x0), epsilon3<scalar_t>());
                        if (function.is_convex())
                        {
                                NANO_CHECK(function.is_convex(x0, x1, 20));
                        }
                }
        }
}

NANO_BEGIN_MODULE(test_functions)

NANO_CASE(evaluate)
{
        foreach_test_function(make_functions(1, 4), test_function);
}

NANO_END_MODULE()
