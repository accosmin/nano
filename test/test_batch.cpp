#include "unit_test.hpp"
#include "math/abs.hpp"
#include "optim/test.hpp"
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

        random_t<scalar_t> rgen(scalar_t(-1), scalar_t(+1));

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
                batch_optimizer::CGD_CD,
                batch_optimizer::CGD_DY,
                batch_optimizer::CGD_FR,
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

                        const auto f_thres = epsilon0<scalar_t>();
                        const auto g_thres = epsilon3<scalar_t>();
                        const auto x_thres = epsilon3<scalar_t>() * scalar_t(1e+3);

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
                                  << ", i = " << state.m_iterations << ".\n";

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
        foreach_test_function<test_type::easy>(1, 4, [] (const auto& function)
        {
                check_function(function);
        });
}

NANO_END_MODULE()

