#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_functions"

#include <boost/test/unit_test.hpp>
#include "core/minimize.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"

#include "func/function_trid.h"
#include "func/function_beale.h"
#include "func/function_booth.h"
#include "func/function_sphere.h"
#include "func/function_matyas.h"
#include "func/function_powell.h"
#include "func/function_mccormick.h"
#include "func/function_himmelblau.h"
#include "func/function_rosenbrock.h"
#include "func/function_3hump_camel.h"
#include "func/function_sum_squares.h"
#include "func/function_dixon_price.h"
#include "func/function_goldstein_price.h"
#include "func/function_rotated_ellipsoid.h"

namespace test
{
        using namespace ncv;

        static void check_function(const std::vector<test::function_t>& funcs)
        {
                BOOST_CHECK_EQUAL(funcs.empty(), false);

                for (const test::function_t& func : funcs)
                {
                        const auto& fn_size = func.m_opsize;
                        const auto& fn_fval = func.m_opfval;
                        const auto& fn_grad = func.m_opgrad;

                        const size_t trials = 1024;

                        const size_t dims = fn_size();
                        BOOST_CHECK_GT(dims, 0);

                        for (size_t t = 0; t < trials; t ++)
                        {
                                random_t<scalar_t> rgen(-1.0, +1.0);

                                vector_t x0(dims);
                                rgen(x0.data(), x0.data() + x0.size());

                                // check gradient
                                const opt_problem_t problem(fn_size, fn_fval, fn_grad);                                
                                BOOST_CHECK_EQUAL(problem.size(), dims);
                                BOOST_CHECK_LE(problem.grad_accuracy(x0), math::epsilon2<scalar_t>());
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
        test::check_function(ncv::make_sphere_funcs(8));
        test::check_function(ncv::make_powell_funcs(32));
        test::check_function(ncv::make_mccormick_funcs());
        test::check_function(ncv::make_himmelblau_funcs());
        test::check_function(ncv::make_rosenbrock_funcs(7));
        test::check_function(ncv::make_3hump_camel_funcs());
        test::check_function(ncv::make_dixon_price_funcs(32));
        test::check_function(ncv::make_sum_squares_funcs(32));
        test::check_function(ncv::make_goldstein_price_funcs());
        test::check_function(ncv::make_rotated_ellipsoid_funcs(32));
}

