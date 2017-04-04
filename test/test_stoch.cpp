#include "utest.h"
#include "math/random.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "functions/test.h"
#include "text/to_string.h"
#include "stoch_optimizer.h"

using namespace nano;

static void check_function(const function_t& function)
{
        const auto epochs = size_t(1000);
        const auto epoch_size = size_t(3000);
        const auto trials = size_t(10);

        const auto dims = function.size();

        auto rgen = make_rng(scalar_t(-1), scalar_t(+1));

        // generate fixed random trials
        std::vector<vector_t> x0s(trials);
        for (auto& x0 : x0s)
        {
                x0.resize(dims);
                rgen(x0.data(), x0.data() + x0.size());
        }

        // optimizers to try
        const auto ids = get_stoch_optimizers().ids();
        for (const auto id : ids)
        {
                if (id == "ngd")
                {
                        // NGD may not decrease the function to epsilon!
                        continue;
                }

                const auto optimizer = get_stoch_optimizers().get(id);

                size_t out_of_domain = 0;

                for (size_t t = 0; t < trials; ++ t)
                {
                        const auto& x0 = x0s[t];
                        const auto f0 = function.eval(x0);
                        const auto g_thres = epsilon3<scalar_t>();

                        // optimize
                        const auto params = stoch_params_t(epochs, epoch_size, epsilon3<scalar_t>());
                        const auto state = optimizer->minimize(params, function, x0);

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
                                  << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                                  << ", f = " << f0 << "/" << f
                                  << ", g = " << g << ".\n";

                        // check function value decrease
                        NANO_CHECK_LESS_EQUAL(f, f0);

                        // check convergence
                        NANO_CHECK_LESS_EQUAL(g, g_thres);
                }

                std::cout << function.name() << ", " << id
                          << ": out of domain " << out_of_domain << "/" << trials << ".\n";
        }
}

NANO_BEGIN_MODULE(test_stoch_optimizers)

NANO_CASE(evaluate)
{
        foreach_test_function(make_convex_functions(1, 1), check_function);
}

NANO_END_MODULE()

