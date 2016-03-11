#include "unit_test.hpp"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "math/funcs/foreach.hpp"

namespace test
{
        template
        <
                typename tscalar,
                typename tvector = typename nano::function_t<tscalar>::tvector
        >
        void test_function(const nano::function_t<tscalar>& function)
        {
                const size_t trials = 135;

                const auto epsilon = nano::epsilon2<tscalar>();

                const auto dims = function.problem().size();
                NANO_CHECK_GREATER(dims, 0);

                for (size_t t = 0; t < trials; ++ t)
                {
                        nano::random_t<tscalar> rgen(tscalar(-0.1), tscalar(+0.1));

                        tvector x0(dims);
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
        nano::foreach_test_function<double, nano::test_type::all>(1, 8, [] (const auto& function)
        {
                test::test_function(function);
        });
}

NANO_END_MODULE()
