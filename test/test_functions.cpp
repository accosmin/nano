#include "unit_test.hpp"
#include "optim/test.hpp"
#include "math/random.hpp"
#include "math/epsilon.hpp"

using namespace nano;

static void test_function(const function_t& function)
{
        const auto trials = size_t(135);
        const auto epsilon = epsilon3<scalar_t>();

        const auto dims = function.problem().size();
        NANO_CHECK_GREATER(dims, 0);

        std::cout << function.name() << std::endl;

        auto rgen = make_rng(scalar_t(-0.1), scalar_t(+0.1));
        for (size_t t = 0; t < trials; ++ t)
        {
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
