#include "unit_test.hpp"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "math/funcs/foreach.hpp"

namespace test
{
        template
        <
                typename tscalar,
                typename tvector = typename zob::function_t<tscalar>::tvector
        >
        void test_function(const zob::function_t<tscalar>& function)
        {
                const size_t trials = 135;

                const auto epsilon = zob::epsilon2<tscalar>();

                const auto dims = function.problem().size();
                ZOB_CHECK_GREATER(dims, 0);

                for (size_t t = 0; t < trials; ++ t)
                {
                        zob::random_t<tscalar> rgen(tscalar(-0.1), tscalar(+0.1));

                        tvector x0(dims);
                        rgen(x0.data(), x0.data() + x0.size());

                        const auto problem = function.problem();
                        ZOB_CHECK_EQUAL(problem.size(), dims);
                        ZOB_CHECK_LESS(problem.grad_accuracy(x0), epsilon);
                }
        }              
}

ZOB_BEGIN_MODULE(test_functions)

ZOB_CASE(evaluate)
{
        zob::foreach_test_function<double, zob::test_type::all>(1, 8, [] (const auto& function)
        {
                test::test_function(function);
        });
}

ZOB_END_MODULE()
