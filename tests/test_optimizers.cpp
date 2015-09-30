#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_optimizers"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "core/logger.h"
#include "core/minimize.h"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"
#include <iomanip>

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

        static void check_function(const test::function_t& func)
        {
                const auto iterations = opt_size_t(8 * 1024);
                const auto epsilon = math::epsilon0<opt_scalar_t>();
                const auto trials = size_t(128);

                const auto dims = func.problem().size();

                math::random_t<opt_scalar_t> rgen(-1.0, +1.0);

                // generate fixed random trials
                std::vector<opt_vector_t> x0s(trials);
                for (auto& x0 : x0s)
                {
                        x0.resize(dims);
                        rgen(x0.data(), x0.data() + x0.size());
                }

                // {optimizers, slack ~ expected convergence rate} to try
                const std::map<min::batch_optimizer, opt_scalar_t> optimizers =
                {
                        { min::batch_optimizer::GD, 1e+3 },             // ok, but potentially very slow!

                        { min::batch_optimizer::CGD, 1e+1 },
                        { min::batch_optimizer::CGD_CD, 1e+1 },
                        { min::batch_optimizer::CGD_DY, 1e+10 },        // bad!
                        { min::batch_optimizer::CGD_FR, 1e+10 },        // bad!
                        { min::batch_optimizer::CGD_HS, 1e+10 },        // bad!
                        { min::batch_optimizer::CGD_LS, 1e+1 },
                        { min::batch_optimizer::CGD_N, 1e+1 },
                        { min::batch_optimizer::CGD_PRP, 1e+1 },
                        { min::batch_optimizer::CGD_DYCD, 1e+1 },
                        { min::batch_optimizer::CGD_DYHS, 1e+1 },

                        { min::batch_optimizer::LBFGS, 1e+1 }
                };

                for (const auto& optslack : optimizers)
                {
                        const auto optimizer = optslack.first;
                        const auto slack = optslack.second;

                        size_t out_of_domain = 0;

                        for (size_t t = 0; t < trials; t ++)
                        {
                                const auto problem = func.problem();

                                const auto& x0 = x0s[t];
                                const auto f0 = problem(x0);

                                // optimize
                                const auto state = ncv::minimize(
                                        problem, nullptr,
                                        x0, optimizer, iterations, epsilon);

                                const auto x = state.x;
                                const auto f = state.f;
                                const auto g = state.convergence_criteria();

                                const auto f_thres = f0 - epsilon * math::abs(f0);
                                const auto g_thres = math::epsilon3<opt_scalar_t>() * slack;
                                const auto x_thres = math::epsilon3<opt_scalar_t>() * slack * 1e+2;

                                // ignore out-of-domain solutions
                                if (!func.is_valid(x))
                                {
                                        out_of_domain ++;
                                        continue;
                                }

                                #define NANOCV_TEST_OPTIMIZERS_DESCRIPTION \
                                        "for (" << func.name() << ", " << text::to_string(optimizer) << \
                                        std::setprecision(12) << \
                                        ", x = [" << x0.transpose() << "]/[" << x.transpose() << "]" << \
                                        ", f = " << f0 << "/" << f << \
                                        ", g = " << g << \
                                        ", i = " << state.m_iterations << ")"

                                // check function value decrease
                                BOOST_CHECK_MESSAGE(f < f0,
                                        "decrease failed " << NANOCV_TEST_OPTIMIZERS_DESCRIPTION);
                                BOOST_CHECK_MESSAGE(f < f_thres,
                                        "sufficient decrease failed " << NANOCV_TEST_OPTIMIZERS_DESCRIPTION);

                                // check convergence
                                BOOST_CHECK_MESSAGE(g < g_thres,
                                        "convergence failed " << NANOCV_TEST_OPTIMIZERS_DESCRIPTION);
//                                BOOST_CHECK_MESSAGE(state.m_status == min::status::converged,
//                                        text::to_string(state.m_status) << " " << NANOCV_TEST_OPTIMIZERS_DESCRIPTION);

                                // check local minimas (if any known)
                                BOOST_CHECK_MESSAGE(func.is_minima(x, x_thres),
                                        "invalid minima " << NANOCV_TEST_OPTIMIZERS_DESCRIPTION);
                        }

                        log_info() << "out of domain for (" << func.name() << ", " << text::to_string(optimizer)
                                   << "): " << out_of_domain << "/" << trials << ".";
                }
        }

        static void check_function(const functions_t& funcs)
        {
                for (const auto& func : funcs)
                {
                        test::check_function(*func);
                }
        }
}

BOOST_AUTO_TEST_CASE(test_optimizers)
{
        using namespace ncv;

        test::check_function(ncv::make_beale_funcs());
        test::check_function(ncv::make_booth_funcs());
        test::check_function(ncv::make_matyas_funcs());
        test::check_function(ncv::make_trid_funcs(8));
        test::check_function(ncv::make_cauchy_funcs(8));
        test::check_function(ncv::make_sphere_funcs(8));
        test::check_function(ncv::make_powell_funcs(8));
        test::check_function(ncv::make_colville_funcs());
        test::check_function(ncv::make_zakharov_funcs(8));
        test::check_function(ncv::make_mccormick_funcs());
        test::check_function(ncv::make_himmelblau_funcs());
        test::check_function(ncv::make_bohachevsky_funcs());
        test::check_function(ncv::make_rosenbrock_funcs(7));
        test::check_function(ncv::make_3hump_camel_funcs());
        test::check_function(ncv::make_dixon_price_funcs(8));
        test::check_function(ncv::make_sum_squares_funcs(8));
        test::check_function(ncv::make_goldstein_price_funcs());
        test::check_function(ncv::make_styblinski_tang_funcs(8));
        test::check_function(ncv::make_rotated_ellipsoid_funcs(8));
}

