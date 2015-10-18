#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_stoch_optimizers"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "math/random.hpp"
#include "cortex/logger.h"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "cortex/minimize.h"
#include "text/to_string.hpp"
#include "min/func/run_all.hpp"

namespace test
{
        using namespace cortex;

        template <typename tfunction>
        static void check_function(const tfunction& func)
        {
                const auto epochs = opt_size_t(128);
                const auto epoch_size = opt_size_t(64);
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
                                const auto problem = func.problem();

                                const auto& x0 = x0s[t];
                                const auto f0 = problem(x0);

                                // optimize
                                opt_scalar_t alpha0, decay;
                                cortex::tune_stochastic(problem, x0, optimizer, epoch_size, alpha0, decay);

                                const auto state = min::minimize(
                                        problem, nullptr, x0, optimizer, epochs, epoch_size, alpha0, decay);

                                const auto x = state.x;
                                const auto f = state.f;
                                const auto g = state.convergence_criteria();

                                const auto f_thres = math::epsilon0<opt_scalar_t>();
                                const auto g_thres = math::epsilon3<opt_scalar_t>() * 1e+3;
                                const auto x_thres = math::epsilon3<opt_scalar_t>() * 1e+3;

                                // ignore out-of-domain solutions
                                if (!func.is_valid(x))
                                {
                                        out_of_domain ++;
                                        continue;
                                }

                                cortex::log_info()
                                        << func.name() << ", " << text::to_string(optimizer)
                                        << " [" << (t + 1) << "/" << trials << "]"
                                        << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                                        << ", f = " << f0 << "/" << f
                                        << ", g = " << g
                                        << ", alpha0 = " << alpha0
                                        << ", decay = " << decay << ".";

                                // check function value decrease
                                BOOST_CHECK_LE(f, f0);
                                BOOST_CHECK_LE(f, f0 - f_thres * math::abs(f0));

                                // check convergence
                                BOOST_CHECK_LE(g, g_thres);

                                // check local minimas (if any known)
                                BOOST_CHECK(func.is_minima(x, x_thres));
                        }

                        cortex::log_info()
                                << func.name() << ", " << text::to_string(optimizer)
                                << ": out of domain " << out_of_domain << "/" << trials << ".";
                }
        }
}

BOOST_AUTO_TEST_CASE(test_stoch_optimizers)
{
        func::run_all_test_functions<cortex::opt_scalar_t>(8, [] (const auto& function)
        {
                test::check_function(function);
        });
}

