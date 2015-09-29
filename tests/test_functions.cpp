#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_functions"

#include <boost/test/unit_test.hpp>
#include "core/minimize.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"

#include "func/function_trid.h"
#include "func/function_beale.h"
#include "func/function_booth.h"
#include "func/function_cauchy.h"
#include "func/function_sphere.h"
#include "func/function_matyas.h"
#include "func/function_powell.h"
#include "func/function_colville.h"
#include "func/function_zakharov.h"
#include "func/function_mccormick.h"
#include "func/function_himmelblau.h"
#include "func/function_rosenbrock.h"
#include "func/function_3hump_camel.h"
#include "func/function_sum_squares.h"
#include "func/function_dixon_price.h"
#include "func/function_bohachevsky.h"
#include "func/function_goldstein_price.h"
#include "func/function_styblinski_tang.h"
#include "func/function_rotated_ellipsoid.h"

namespace test
{
        using namespace ncv;

        static void check_function(const functions_t& funcs)
        {
                BOOST_CHECK_EQUAL(funcs.empty(), false);

                for (const rfunction_t& func : funcs)
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
}

BOOST_AUTO_TEST_CASE(test_functions)
{
        test::check_function(ncv::make_beale_funcs());
        test::check_function(ncv::make_booth_funcs());
        test::check_function(ncv::make_matyas_funcs());
        test::check_function(ncv::make_trid_funcs(32));
        test::check_function(ncv::make_cauchy_funcs(8));
        test::check_function(ncv::make_sphere_funcs(8));        
        test::check_function(ncv::make_powell_funcs(32));
        test::check_function(ncv::make_colville_funcs());
        test::check_function(ncv::make_zakharov_funcs(8));
        test::check_function(ncv::make_mccormick_funcs());
        test::check_function(ncv::make_himmelblau_funcs());
        test::check_function(ncv::make_bohachevsky_funcs());
        test::check_function(ncv::make_rosenbrock_funcs(7));
        test::check_function(ncv::make_3hump_camel_funcs());
        test::check_function(ncv::make_dixon_price_funcs(32));
        test::check_function(ncv::make_sum_squares_funcs(32));
        test::check_function(ncv::make_goldstein_price_funcs());
        test::check_function(ncv::make_styblinski_tang_funcs(32));
        test::check_function(ncv::make_rotated_ellipsoid_funcs(32));
}

