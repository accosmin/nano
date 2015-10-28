#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_stoch_optimizers"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "min/tune_stoch.hpp"
#include "text/to_string.hpp"
#include "cortex/optimizer.h"
#include "min/test/run_all.hpp"
#include <iostream>

namespace test
{
        template
        <
                typename tscalar,
                typename tvector = typename min::function_t<tscalar>::tvector
        >
        static void check_function(const min::function_t<tscalar>& function)
        {
                const auto epochs = size_t(128);
                const auto epoch_size = size_t(64);
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

                // optimizers to try
                const auto optimizers =
                {
                        min::stoch_optimizer::SG,
                        min::stoch_optimizer::SGA,
                        min::stoch_optimizer::SIA,
                        min::stoch_optimizer::AG,
                        min::stoch_optimizer::AGGR,
                        min::stoch_optimizer::ADAGRAD,
                        min::stoch_optimizer::ADADELTA
                };

                for (const auto optimizer : optimizers)
                {
                        size_t out_of_domain = 0;

                        for (size_t t = 0; t < trials; t ++)
                        {
                                const auto problem = function.problem();

                                const auto& x0 = x0s[t];
                                const auto f0 = problem(x0);

                                // optimize
                                tscalar alpha, decay;
                                min::tune_stochastic(problem, x0, optimizer, epoch_size, alpha, decay);

                                const auto state = min::minimize(
                                        problem, nullptr, x0, optimizer, epochs, epoch_size, alpha, decay);

                                const auto x = state.x;
                                const auto f = state.f;
                                const auto g = state.convergence_criteria();

                                const auto f_thres = math::epsilon3<tscalar>();
//                                const auto g_thres = math::epsilon3<tscalar>() * 1e+3;
//                                const auto x_thres = math::epsilon3<tscalar>() * 1e+4;

                                // ignore out-of-domain solutions
                                if (!function.is_valid(x))
                                {
                                        out_of_domain ++;
                                        continue;
                                }

                                std::cout << function.name() << ", " << text::to_string(optimizer)
                                          << " [" << (t + 1) << "/" << trials << "]"
                                          << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                                          << ", f = " << f0 << "/" << f
                                          << ", g = " << g
                                          << ", alpha = " << alpha
                                          << ", decay = " << decay << ".\n";

                                // check function value decrease
                                BOOST_CHECK_LE(f, f0);
                                BOOST_CHECK_LE(f, f0 - f_thres * math::abs(f0));

                                // check convergence
//                                BOOST_CHECK_LE(g, g_thres);

                                // check local minimas (if any known)
//                                BOOST_CHECK(function.is_minima(x, x_thres));
                        }

                        std::cout << function.name() << ", " << text::to_string(optimizer)
                                  << ": out of domain " << out_of_domain << "/" << trials << "." << std::endl;
                }
        }
}

BOOST_AUTO_TEST_CASE(test_stoch_optimizers)
{
        min::run_all_test_functions<double>(8, [] (const auto& function)
        {
                test::check_function(function);
        });
}

