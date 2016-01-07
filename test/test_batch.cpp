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

namespace test
{
        template
        <
                typename tscalar,
                typename tvector = typename math::function_t<tscalar>::tvector
        >
        static void check_function(const math::function_t<tscalar>& function)
        {
                const auto iterations = size_t(1024);
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
                        math::batch_optimizer::GD,

                        math::batch_optimizer::CGD,
//                        math::batch_optimizer::CGD_CD,
//                        math::batch_optimizer::CGD_DY,
//                        math::batch_optimizer::CGD_FR,
//                        math::batch_optimizer::CGD_HS,
//                        math::batch_optimizer::CGD_LS,
//                        math::batch_optimizer::CGD_N,
//                        math::batch_optimizer::CGD_PRP,
//                        math::batch_optimizer::CGD_DYCD,
//                        math::batch_optimizer::CGD_DYHS,

                        math::batch_optimizer::LBFGS,
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
                                const auto state = math::minimize(
                                        problem, nullptr, x0, optimizer, iterations, math::epsilon0<tscalar>());

                                const auto x = state.x;
                                const auto f = state.f;
                                const auto g = state.convergence_criteria();

                                const auto f_thres = math::epsilon0<tscalar>();
                                const auto g_thres = math::epsilon3<tscalar>();
                                const auto x_thres = math::epsilon3<tscalar>() * 1e+3;

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
                                NANOCV_CHECK_LESS(f, f0);
                                NANOCV_CHECK_LESS(f, f0 - f_thres * math::abs(f0));

                                // check convergence
                                NANOCV_CHECK_LESS(g, g_thres);

                                // check local minimas (if any known)
                                NANOCV_CHECK(function.is_minima(x, x_thres));
                        }

                        std::cout << function.name() << ", " << text::to_string(optimizer)
                                  << ": out of domain " << out_of_domain << "/" << trials << ".\n";
                }
        }
}

NANOCV_BEGIN_MODULE(test_batch_optimizers)

NANOCV_CASE(evaluate)
{
        math::foreach_test_function<double, math::test_type::easy>(1, 4, [] (const math::function_t<double>& function)
        {
                test::check_function(function);
        });
}

NANOCV_END_MODULE()

