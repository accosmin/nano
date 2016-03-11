#include "unit_test.hpp"
#include "math/abs.hpp"
#include "math/batch.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"
#include "cortex/optimizer.h"
#include "math/funcs/foreach.hpp"
#include <iomanip>

template
<
        typename tscalar,
        typename tvector = typename nano::function_t<tscalar>::tvector
>
static void check_function(const nano::function_t<tscalar>& function)
{
        const auto iterations = size_t(1024);
        const auto trials = size_t(32);

        const auto dims = function.problem().size();

        nano::random_t<tscalar> rgen(tscalar(-1), tscalar(+1));

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
                nano::batch_optimizer::GD,

                nano::batch_optimizer::CGD,
                nano::batch_optimizer::CGD_CD,
                nano::batch_optimizer::CGD_DY,
                nano::batch_optimizer::CGD_FR,
                //nano::batch_optimizer::CGD_HS,
                nano::batch_optimizer::CGD_LS,
                nano::batch_optimizer::CGD_N,
                nano::batch_optimizer::CGD_PRP,
                nano::batch_optimizer::CGD_DYCD,
                nano::batch_optimizer::CGD_DYHS,

                nano::batch_optimizer::LBFGS,
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
                        const auto state = nano::minimize(
                                problem, nullptr, x0, optimizer, iterations, nano::epsilon0<tscalar>());

                        const auto x = state.x;
                        const auto f = state.f;
                        const auto g = state.convergence_criteria();

                        const auto f_thres = nano::epsilon0<tscalar>();
                        const auto g_thres = nano::epsilon3<tscalar>();
                        const auto x_thres = nano::epsilon3<tscalar>() * 1e+3;

                        // ignore out-of-domain solutions
                        if (!function.is_valid(x))
                        {
                                out_of_domain ++;
                                continue;
                        }

                        std::cout << function.name() << ", " << nano::to_string(optimizer)
                                  << " [" << (t + 1) << "/" << trials << "]"
                                  << std::setprecision(12)
                                  << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                                  << ", f = " << f0 << "/" << f
                                  << ", g = " << g
                                  << ", i = " << state.m_iterations << ".\n";

                        // check function value decrease
                        NANO_CHECK_LESS(f, f0);
                        NANO_CHECK_LESS(f, f0 - f_thres * nano::abs(f0));

                        // check convergence
                        NANO_CHECK_LESS(g, g_thres);

                        // check local minimas (if any known)
                        NANO_CHECK(function.is_minima(x, x_thres));
                }

                std::cout << function.name() << ", " << nano::to_string(optimizer)
                          << ": out of domain " << out_of_domain << "/" << trials << ".\n";
        }
}

NANO_BEGIN_MODULE(test_batch_optimizers)

NANO_CASE(evaluate)
{
        nano::foreach_test_function<double, nano::test_type::easy>(1, 4, [] (const auto& function)
        {
                check_function(function);
        });
}

NANO_END_MODULE()

