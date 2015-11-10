#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_batch_optimizers"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "math/batch.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"
#include "cortex/optimizer.h"
#include "math/funcs/run_all.hpp"
#include <iomanip>
#include <iostream>

namespace test
{
        template
        <
                typename tscalar,
                typename tvector = typename math::function_t<tscalar>::tvector
        >
        static void check_function(const math::function_t<tscalar>& function)
        {
                const auto iterations = size_t(8 * 1024);
                const auto trials = size_t(32);

                const auto dims = function.problem().size();

                math::random_t<tscalar> rgen(tscalar(-1), tscalar(+1));

                // generate fixed random trials
                std::vector<tvector> x0s(trials);
                for (auto& x0 : x0s)
                {
                        x0.resize(dims);
                        rgen(x0.data(), x0.data() + x0.size());
                }

                // {optimizers, slack ~ expected convergence rate} to try
                const std::map<math::batch_optimizer, tscalar> optimizers =
                {
                        { math::batch_optimizer::GD, 1e+3 },             // ok, but potentially very slow!

                        { math::batch_optimizer::CGD, 1e+0 },
                        { math::batch_optimizer::CGD_CD, 1e+0 },
                        { math::batch_optimizer::CGD_DY, 1e+10 },        // bad!
                        { math::batch_optimizer::CGD_FR, 1e+10 },        // bad!
                        { math::batch_optimizer::CGD_HS, 1e+10 },        // bad!
                        { math::batch_optimizer::CGD_LS, 1e+0 },
                        { math::batch_optimizer::CGD_N, 1e+0 },
                        { math::batch_optimizer::CGD_PRP, 1e+0 },
                        { math::batch_optimizer::CGD_DYCD, 1e+0 },
                        { math::batch_optimizer::CGD_DYHS, 1e+0 },

                        { math::batch_optimizer::LBFGS, 1e+0 }
                };

                for (const auto& optslack : optimizers)
                {
                        const auto optimizer = optslack.first;
                        const auto slack = optslack.second;

                        size_t out_of_domain = 0;

                        for (size_t t = 0; t < trials; ++ t)
                        {
                                const auto problem = function.problem();

                                const auto& x0 = x0s[t];
                                const auto f0 = problem(x0);

                                // optimize
                                const auto state = math::minimize(
                                        problem, nullptr, x0, optimizer, iterations, math::epsilon0<tscalar>());

                                const auto x = state.x;
                                const auto f = state.f;
                                const auto g = state.convergence_criteria();

                                const auto f_thres = math::epsilon0<tscalar>();
                                const auto g_thres = math::epsilon3<tscalar>() * slack;
                                const auto x_thres = math::epsilon3<tscalar>() * slack * 1e+3;

                                // ignore out-of-domain solutions
                                if (!function.is_valid(x))
                                {
                                        out_of_domain ++;
                                        continue;
                                }

                                std::cout << function.name() << ", " << text::to_string(optimizer)
                                          << " [" << (t + 1) << "/" << trials << "]"
                                          << std::setprecision(12)
                                          << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                                          << ", f = " << f0 << "/" << f
                                          << ", g = " << g
                                          << ", i = " << state.m_iterations << ".\n";

                                // check function value decrease
                                BOOST_CHECK_LE(f, f0);
                                BOOST_CHECK_LE(f, f0 - f_thres * math::abs(f0));

                                // check convergence
                                BOOST_CHECK_LE(g, g_thres);

                                // check local minimas (if any known)
                                BOOST_CHECK(function.is_minima(x, x_thres));
                        }

                        std::cout << function.name() << ", " << text::to_string(optimizer)
                                  << ": out of domain " << out_of_domain << "/" << trials << "." << std::endl;
                }
        }
}

BOOST_AUTO_TEST_CASE(test_batch_optimizers)
{
        math::run_all_test_functions<double>(8, [] (const auto& function)
        {
                test::check_function(function);
        });
}

