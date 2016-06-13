#include "unit_test.hpp"
#include "optim/test.hpp"
#include "optim/stoch.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"

using namespace nano;

static void check_function(const function_t& function)
{
        const auto epochs = size_t(32);
        const auto epoch_size = size_t(16);
        const auto trials = size_t(16);

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
                stoch_optimizer::SG,
//                stoch_optimizer::NGD,
                stoch_optimizer::SGM,
                stoch_optimizer::AG,
                stoch_optimizer::AGFR,
                stoch_optimizer::AGGR,
                stoch_optimizer::ADAGRAD,
                stoch_optimizer::ADADELTA,
                stoch_optimizer::ADAM
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
                        const auto state = minimize(problem, nullptr, x0, optimizer, epochs, epoch_size);

                        const auto x = state.x;
                        const auto f = state.f;
                        const auto g = state.convergence_criteria();

                        const auto f_thres = epsilon3<scalar_t>();
                        const auto g_thres = std::cbrt(epsilon3<scalar_t>());
                        const auto x_thres = std::sqrt(epsilon3<scalar_t>());

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
        foreach_test_function<test_type::easy>(1, 4, [] (const auto& function)
        {
                check_function(function);
        });
}

NANO_END_MODULE()

