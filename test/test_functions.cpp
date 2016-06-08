#include "unit_test.hpp"
#include "optim/test.hpp"
#include "math/random.hpp"
#include "math/epsilon.hpp"

using namespace nano;

static void test_function(const function_t& function)
{
        const auto trials = size_t(135);
        const auto epsilon = 10 * epsilon3<scalar_t>();

        const auto dims = function.problem().size();
        NANO_CHECK_GREATER(dims, 0);

        for (size_t t = 0; t < trials; ++ t)
        {
                random_t<scalar_t> rgen(scalar_t(-0.01), scalar_t(+0.01));

                vector_t x0(dims);
                rgen(x0.data(), x0.data() + x0.size());

                const auto problem = function.problem();
                NANO_CHECK_EQUAL(problem.size(), dims);
                NANO_CHECK_LESS(problem.grad_accuracy(x0), epsilon);
        }
}

NANO_BEGIN_MODULE(test_functions)

NANO_CASE(evaluate)
{
        foreach_test_function<test_type::all>(1, 8, [] (const function_t& function)
        {
                test_function(function);
        });
}

NANO_END_MODULE()
