#include "utest.h"
#include "function.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "solver_stoch.h"

using namespace nano;

static void check_function(const function_t& function)
{
        const auto epochs = size_t(1000);
        const auto tune_epochs = size_t(100);
        const auto trials = size_t(10);

        // generate fixed random trials
        std::vector<vector_t> x0s(trials);
        for (auto& x0 : x0s)
        {
                x0 = vector_t::Random(function.size());
        }

        // solvers to try
        for (const auto& id : get_stoch_solvers().ids())
        {
                const auto solver = get_stoch_solvers().get(id);
                NANO_REQUIRE(solver);

                size_t out_of_domain = 0;
                for (size_t t = 0; t < trials; ++ t)
                {
                        const auto& x0 = x0s[t];
                        const auto f0 = function.eval(x0);
                        const auto g_thres = epsilon3<scalar_t>();

                        // optimize
                        const auto params = stoch_params_t(epochs, epsilon2<scalar_t>());
                        const auto tune_params = stoch_params_t(tune_epochs, epsilon2<scalar_t>());

                        const auto tstate = solver->tune(tune_params, function, x0);
                        const auto state = solver->minimize(params, function, tstate.x);

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
                                  << ": x=[" << x0.transpose() << "]/[" << x.transpose() << "]"
                                  << ",f=" << f0 << "/" << f
                                  << ",g=" << g
                                  << ",calls=" << function.fcalls() << "/" << function.gcalls() << ".\n";

                        // check function value decrease
                        NANO_CHECK_LESS_EQUAL(f, f0);

                        // check convergence
                        NANO_CHECK_LESS_EQUAL(g, g_thres);
                }

                std::cout << function.name() << ", " << id
                          << ": out of domain " << out_of_domain << "/" << trials << ".\n";
        }
}

NANO_BEGIN_MODULE(test_stoch_solvers)

NANO_CASE(evaluate)
{
        for (const auto& function : get_convex_functions(1, 1))
        {
                check_function(*function);
        }
}

NANO_END_MODULE()

