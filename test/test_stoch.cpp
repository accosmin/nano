#include "unit_test.hpp"
#include "optim/test.h"
#include "optim/stoch.h"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"

using namespace nano;

static void check_function(const function_t& function)
{
        const auto epochs = size_t(100);
        const auto epoch_size = size_t(10);
        const auto trials = size_t(20);

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
        for (const auto optimizer : enum_values<stoch_optimizer>())
        {
                size_t out_of_domain = 0;

                for (size_t t = 0; t < trials; ++ t)
                {
                        const auto problem = function.problem();

                        const auto& x0 = x0s[t];
                        const auto f0 = problem(x0);

                        // optimize
                        const auto params = stoch_params_t(epochs, epoch_size, optimizer);
                        const auto state = minimize(params, problem, x0);

                        const auto x = state.x;
                        const auto f = state.f;
                        const auto g = state.convergence_criteria();

                        const auto f_thres = epsilon3<scalar_t>();
                        const auto g_thres = std::cbrt(epsilon3<scalar_t>());
                        const auto x_thres = std::cbrt(epsilon3<scalar_t>()) * scalar_t(10);

                        // ignore out-of-domain solutions
                        if (!function.is_valid(x))
                        {
                                out_of_domain ++;
                                continue;
                        }

                        std::cout << function.name() << ", " << to_string(optimizer)
                                  << " [" << (t + 1) << "/" << trials << "]"
                                  << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                                  << ", f = " << f0 << "/" << f
                                  << ", g = " << g << ".\n";

                        // check function value decrease
                        NANO_CHECK_LESS_EQUAL(f, f0);
                        NANO_CHECK_LESS_EQUAL(f, f0 - f_thres * abs(f0));

                        // check convergence
                        NANO_CHECK_LESS_EQUAL(g, g_thres);

                        // check local minimas (if any known)
                        NANO_CHECK(function.is_minima(x, x_thres));
                }

                std::cout << function.name() << ", " << to_string(optimizer)
                          << ": out of domain " << out_of_domain << "/" << trials << ".\n";
        }
}

NANO_BEGIN_MODULE(test_stoch_optimizers)

NANO_CASE(evaluate)
{
        foreach_test_function(make_convex_functions(1, 4), check_function);
}

NANO_END_MODULE()

