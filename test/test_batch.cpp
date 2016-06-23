#include "unit_test.hpp"
#include "math/abs.hpp"
#include "optim/test.h"
#include "optim/batch.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"
#include <iomanip>

using namespace nano;

static void check_function(const function_t& function)
{
        const auto iterations = size_t(1024);
        const auto trials = size_t(32);

        const auto dims = function.problem().size();

        auto rgen = make_rng(scalar_t(-1), scalar_t(+1));

        // generate fixed random trials
        std::vector<vector_t> x0s(trials);
        for (auto& x0 : x0s)
        {
                x0.resize(dims);
                rgen(x0.data(), x0.data() + x0.size());
        }

        // optimizers to try
        const auto optimizers =
        {
                batch_optimizer::GD,

                batch_optimizer::CGD,
                //batch_optimizer::CGD_CD,
                //batch_optimizer::CGD_DY,
                //batch_optimizer::CGD_FR,
                //batch_optimizer::CGD_HS,
                batch_optimizer::CGD_LS,
                batch_optimizer::CGD_N,
                batch_optimizer::CGD_PRP,
                batch_optimizer::CGD_DYCD,
                batch_optimizer::CGD_DYHS,

                batch_optimizer::LBFGS,
        };

        for (const auto& optimizer : optimizers)
        {
                size_t out_of_domain = 0;

                for (size_t t = 0; t < trials; ++ t)
                {
                        const auto problem = function.problem();

                        const auto& x0 = x0s[t];
                        const auto f0 = problem(x0);

                        // optimize
                        const auto state = minimize(
                                problem, nullptr, x0, optimizer, iterations, epsilon0<scalar_t>());

                        const auto x = state.x;
                        const auto f = state.f;
                        const auto g = state.convergence_criteria();

                        const auto f_thres = epsilon3<scalar_t>();
                        const auto g_thres = std::sqrt(epsilon3<scalar_t>());
                        const auto x_thres = std::sqrt(epsilon3<scalar_t>());

                        // ignore out-of-domain solutions
                        if (!function.is_valid(x))
                        {
                                out_of_domain ++;
                                continue;
                        }

                        std::cout << function.name() << ", " << to_string(optimizer)
                                  << " [" << (t + 1) << "/" << trials << "]"
                                  << std::setprecision(12)
                                  << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                                  << ", f = " << f0 << "/" << f
                                  << ", g = " << g
                                  << ", i = " << state.m_iterations
                                  << ", status = " << to_string(state.m_status) << ".\n";

                        NANO_CHECK(state.m_status == state_t::status::converged);

                        // check function value decrease
                        NANO_CHECK_LESS(f, f0);
                        NANO_CHECK_LESS(f, f0 - f_thres * abs(f0));

                        // check convergence
                        NANO_CHECK_LESS(g, g_thres);

                        // check local minimas (if any known)
                        NANO_CHECK(function.is_minima(x, x_thres));
                }

                std::cout << function.name() << ", " << to_string(optimizer)
                          << ": out of domain " << out_of_domain << "/" << trials << ".\n";
        }
}

NANO_BEGIN_MODULE(test_batch_optimizers)

NANO_CASE(evaluate)
{
        foreach_test_function(make_convex_functions(1, 4), check_function);
}

NANO_END_MODULE()

