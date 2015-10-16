#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_batch_optimizers"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "cortex/logger.h"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "cortex/minimize.h"
#include "text/to_string.hpp"
#include "minfunc/make_functions.hpp"
#include <iomanip>

namespace test
{
        using namespace ncv;

        template <typename tfunction>
        static void check_function(const tfunction& func)
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
                                const auto state = min::minimize(
                                        problem, nullptr, x0, optimizer, iterations, epsilon);

                                const auto x = state.x;
                                const auto f = state.f;
                                const auto g = state.convergence_criteria();

                                const auto f_thres = math::epsilon0<opt_scalar_t>();
                                const auto g_thres = math::epsilon3<opt_scalar_t>() * slack;
                                const auto x_thres = math::epsilon3<opt_scalar_t>() * slack * 1e+1;

                                // ignore out-of-domain solutions
                                if (!func.is_valid(x))
                                {
                                        out_of_domain ++;
                                        continue;
                                }

                                ncv::log_info()
                                        << func.name() << ", " << text::to_string(optimizer)
                                        << " [" << (t + 1) << "/" << trials << "]"
                                        << std::setprecision(12)
                                        << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                                        << ", f = " << f0 << "/" << f
                                        << ", g = " << g
                                        << ", i = " << state.m_iterations << ".";

                                // check function value decrease
                                BOOST_CHECK_LE(f, f0);
                                BOOST_CHECK_LE(f, f0 - f_thres * math::abs(f0));

                                // check convergence
                                BOOST_CHECK_LE(g, g_thres);

                                // check local minimas (if any known)
                                BOOST_CHECK(func.is_minima(x, x_thres));
                        }

                        ncv::log_info()
                                << func.name() << ", " << text::to_string(optimizer)
                                << ": out of domain " << out_of_domain << "/" << trials << ".";
                }
        }
}

BOOST_AUTO_TEST_CASE(test_batch_optimizers)
{
        const auto funcs = func::make_all_test_functions<ncv::opt_scalar_t>(8);
        for (const auto& func : funcs)
        {
                test::check_function(*func);
        }
}

