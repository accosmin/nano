#include "unit_test.hpp"
#include "math/stoch.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"
#include "cortex/optimizer.h"
#include "math/funcs/foreach.hpp"

template
<
        typename tscalar,
        typename tvector = typename nano::function_t<tscalar>::tvector
>
static void check_function(const nano::function_t<tscalar>& function)
{
        const auto epochs = size_t(128);
        const auto epoch_size = size_t(32);
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
                nano::stoch_optimizer::SG,
//                nano::stoch_optimizer::SNG,
                nano::stoch_optimizer::SGM,
                nano::stoch_optimizer::AG,
                nano::stoch_optimizer::AGFR,
                nano::stoch_optimizer::AGGR,
                nano::stoch_optimizer::ADAGRAD,
                nano::stoch_optimizer::ADADELTA,
                nano::stoch_optimizer::ADAM
        };

        for (const auto optimizer : optimizers)
        {
                size_t out_of_domain = 0;

                for (size_t t = 0; t < trials; ++ t)
                {
                        const auto problem = function.problem();

                        const auto& x0 = x0s[t];
                        const auto f0 = problem(x0);

                        // optimize
                        const auto state = nano::minimize(problem, nullptr, nullptr, x0, optimizer, epochs, epoch_size);

                        const auto x = state.x;
                        const auto f = state.f;
                        const auto g = state.convergence_criteria();

                        const auto f_thres = nano::epsilon3<tscalar>();
                        const auto g_thres = nano::epsilon3<tscalar>() * 1e+4;
                        const auto x_thres = nano::epsilon3<tscalar>() * 1e+4;

                        // ignore out-of-domain solutions
                        if (!function.is_valid(x))
                        {
                                out_of_domain ++;
                                continue;
                        }

                        std::cout << function.name() << ", " << nano::to_string(optimizer)
                                  << " [" << (t + 1) << "/" << trials << "]"
                                  << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                                  << ", f = " << f0 << "/" << f
                                  << ", g = " << g << ".\n";

                        // check function value decrease
                        NANO_CHECK_LESS_EQUAL(f, f0);
                        NANO_CHECK_LESS_EQUAL(f, f0 - f_thres * nano::abs(f0));

                        // check convergence
                        NANO_CHECK_LESS_EQUAL(g, g_thres);

                        // check local minimas (if any known)
                        NANO_CHECK(function.is_minima(x, x_thres));
                }

                std::cout << function.name() << ", " << nano::to_string(optimizer)
                          << ": out of domain " << out_of_domain << "/" << trials << ".\n";
        }
}

NANO_BEGIN_MODULE(test_stoch_optimizers)

NANO_CASE(evaluate)
{
        nano::foreach_test_function<double, nano::test_type::easy>(1, 4, [] (const auto& function)
        {
                check_function(function);
        });
}

NANO_END_MODULE()

