#include "unit_test.hpp"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "optim/funcs/foreach.hpp"

namespace test
{
        using namespace nano;

        void test_function(const function_t& function)
        {
                const auto trials = size_t(135);
                const auto epsilon = epsilon2<tscalar>();

                const auto dims = function.problem().size();
                NANO_CHECK_GREATER(dims, 0);

                for (size_t t = 0; t < trials; ++ t)
                {
                        random_t<scalar_t> rgen(scalar_t(-0.1), scalar_t(+0.1));

                        vector_t x0(dims);
                        rgen(x0.data(), x0.data() + x0.size());

                        const auto problem = function.problem();
                        NANO_CHECK_EQUAL(problem.size(), dims);
                        NANO_CHECK_LESS(problem.grad_accuracy(x0), epsilon);
                }
        }
}

NANO_BEGIN_MODULE(test_functions)

NANO_CASE(evaluate)
{
        nano::foreach_test_function<nano::test_type::all>(1, 8, [] (const nano::function_t& function)
        {
                test::test_function(function);
        });
}

NANO_END_MODULE()
