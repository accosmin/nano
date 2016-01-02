#include "unit_test.hpp"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "math/funcs/foreach.hpp"

namespace test
{
        template
        <
                typename tscalar,
                typename tvector = typename math::function_t<tscalar>::tvector
        >
        void test_function(const math::function_t<tscalar>& function)
        {
                const size_t trials = 135;

                const auto epsilon = math::epsilon2<tscalar>();

                const auto dims = function.problem().size();
                NANOCV_CHECK_GREATER(dims, 0);

                for (size_t t = 0; t < trials; ++ t)
                {
                        math::random_t<tscalar> rgen(tscalar(-0.1), tscalar(+0.1));

                        tvector x0(dims);
                        rgen(x0.data(), x0.data() + x0.size());

                        const auto problem = function.problem();
                        NANOCV_CHECK_EQUAL(problem.size(), dims);
                        NANOCV_CHECK_LESS(problem.grad_accuracy(x0), epsilon);
                }
        }              
}

NANOCV_BEGIN_MODULE(test_functions)

NANOCV_CASE(evaluate)
{
        math::foreach_test_function<double, math::test_type::all>(1, 8, [] (const auto& function)
        {
                test::test_function(function);
        });
}

NANOCV_END_MODULE()
