#include "unit_test.hpp"
#include "optim/test.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "tensor/numeric.hpp"

using namespace nano;

static void test_function(const function_t& function)
{
        std::cout << function.name() << std::endl;

        const auto trials = size_t(1000);
        const auto epsilon = epsilon3<scalar_t>();

        const auto dims = function.problem().size();

        NANO_CHECK_GREATER(dims, 0);
        NANO_CHECK_GREATER_EQUAL(dims, function.min_dims());
        NANO_CHECK_GREATER_EQUAL(function.max_dims(), dims);

        auto rgen = make_rng(scalar_t(-1), scalar_t(+1));

        bool is_convex = function.is_convex();
        for (size_t t = 0; t < trials; ++ t)
        {
                vector_t x0(dims), x1(dims);
                tensor::set_random(rgen, x0, x1);

                const auto problem = function.problem();
                NANO_REQUIRE_EQUAL(problem.size(), dims);
//                NANO_CHECK_LESS(problem.grad_accuracy(x0), epsilon);
                is_convex = is_convex && problem.is_convex(x0, x1, 10);
        }

        NANO_CHECK_EQUAL(is_convex, function.is_convex());
}

NANO_BEGIN_MODULE(test_functions)

NANO_CASE(evaluate)
{
        foreach_test_function(make_functions(1, 8), test_function);
}

NANO_END_MODULE()
