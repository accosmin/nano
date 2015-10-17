#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_functions"

#include <boost/test/unit_test.hpp>
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "min/func/run_all.hpp"

namespace test
{
        template
        <
                typename tfunction,
                typename tscalar = typename tfunction::tscalar,
                typename tvector = typename tfunction::tvector
        >
        void test_function(const tfunction& function)
        {
                const size_t trials = 1024;

                const tscalar epsilon = (sizeof(tscalar) == sizeof(float)) ?
                        math::epsilon3<tscalar>() :
                        math::epsilon2<tscalar>();

                const auto dims = function.problem().size();
                BOOST_CHECK_GT(dims, 0);

                for (size_t t = 0; t < trials; t ++)
                {
                        math::random_t<tscalar> rgen(-0.1, +0.1);

                        tvector x0(dims);
                        rgen(x0.data(), x0.data() + x0.size());

                        // check gradient
                        const auto problem = function.problem();
                        BOOST_CHECK_EQUAL(problem.size(), dims);
                        BOOST_CHECK_LE(problem.grad_accuracy(x0), epsilon);
                        BOOST_CHECK_MESSAGE(problem.grad_accuracy(x0) < epsilon,
                                "invalid gradient for [" << function.name() <<
                                "] & [scalar "  << typeid(tscalar).name() << "]!");
                }
        }

        template <typename tscalar>
        void test_functions()
        {
                func::run_all_test_functions<tscalar>(8, [] (const auto& function)
                {
                        test_function(function);
                });
        }                
}

BOOST_AUTO_TEST_CASE(test_functions)
{
        test::test_functions<float>();
        test::test_functions<double>();
        test::test_functions<long double>();
}

