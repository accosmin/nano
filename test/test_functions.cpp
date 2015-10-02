#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_functions"

#include <boost/test/unit_test.hpp>
#include "core/minimize.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "func/make_functions.h"

BOOST_AUTO_TEST_CASE(test_functions)
{
        using namespace ncv;

        const auto funcs = ncv::make_all_test_functions(8);
        BOOST_CHECK_EQUAL(funcs.empty(), false);

        for (const auto& func : funcs)
        {
                const size_t trials = 1024;

                const opt_size_t dims = func->problem().size();
                BOOST_CHECK_GT(dims, 0);

                for (size_t t = 0; t < trials; t ++)
                {
                        math::random_t<opt_scalar_t> rgen(-1.0, +1.0);

                        opt_vector_t x0(dims);
                        rgen(x0.data(), x0.data() + x0.size());

                        // check gradient
                        const opt_problem_t problem = func->problem();
                        BOOST_CHECK_EQUAL(problem.size(), dims);
                        BOOST_CHECK_LE(problem.grad_accuracy(x0), math::epsilon2<scalar_t>());
                        BOOST_CHECK_MESSAGE(problem.grad_accuracy(x0) < math::epsilon2<scalar_t>(),
                                "invalid gradient for the " << func->name() << " function!");
                }
        }
}

