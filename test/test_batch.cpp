#include "utest.h"
#include "math/random.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "text/to_string.h"
#include "functions/test.h"
#include "batch_optimizer.h"
#include <iomanip>

using namespace nano;

static void check_function(const function_t& function)
{
        const auto iterations = size_t(100000);
        const auto trials = size_t(100);

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
        const auto ids = get_batch_optimizers().ids();
        for (const auto id : ids)
        {
                const auto optimizer = get_batch_optimizers().get(id);

                size_t out_of_domain = 0;

                for (size_t t = 0; t < trials; ++ t)
                {
                        const auto problem = function.problem();

                        const auto& x0 = x0s[t];
                        const auto f0 = problem(x0);
                        const auto g_thres = epsilon2<scalar_t>();
                        const auto x_thres = std::cbrt(g_thres);

                        // optimize
                        const auto params = batch_params_t(iterations, g_thres / 2);
                        const auto state = optimizer->minimize(params, problem, x0);

                        const auto x = state.x;
                        const auto f = state.f;
                        const auto g = state.convergence_criteria();

                        // ignore out-of-domain solutions
                        if (!function.is_valid(x))
                        {
                                out_of_domain ++;
                                continue;
                        }

                        std::cout << function.name() << ", " << id
                                  << " [" << (t + 1) << "/" << trials << "]"
                                  << std::setprecision(12)
                                  << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                                  << ", f = " << f0 << "/" << f
                                  << ", g = " << g
                                  << ", i = " << state.m_iterations
                                  << ", status = " << to_string(state.m_status) << ".\n";

                        // check function value decrease
                        NANO_CHECK_LESS_EQUAL(f, f0);

                        // check convergence
                        NANO_CHECK(state.m_status == opt_status::converged || g < g_thres);
                        NANO_CHECK_LESS(g, g_thres);

                        // check local minimas (if any known)
                        NANO_CHECK(function.is_minima(x, x_thres));
                }

                std::cout << function.name() << ", " << id
                          << ": out of domain " << out_of_domain << "/" << trials << ".\n";
        }
}

NANO_BEGIN_MODULE(test_batch_optimizers)

NANO_CASE(evaluate)
{
        foreach_test_function(make_convex_functions(1, 4), check_function);
}

NANO_END_MODULE()

