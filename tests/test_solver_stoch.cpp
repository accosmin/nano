#include "utest.h"
#include "function.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "solver_stoch.h"

using namespace nano;

namespace nano
{
        std::ostream& operator<<(std::ostream& os, const opt_status status)
        {
                return os << to_string(status);
        }
}

static void check_function(const function_t& function)
{
        const auto epochs = size_t(1000);
        const auto trials = size_t(10);

        // solvers to try
        for (const auto& id : get_stoch_solvers().ids())
        {
                const auto solver = get_stoch_solvers().get(id);
                NANO_REQUIRE(solver);

                size_t out_of_domain = 0;
                for (size_t t = 0; t < trials; ++ t)
                {
                        const auto x0 = vector_t::Random(function.size());
                        const auto f0 = function.eval(x0);
                        const auto fcalls0 = function.fcalls();
                        const auto gcalls0 = function.gcalls();

                        // optimize
                        const auto params = stoch_params_t(epochs, epsilon2<scalar_t>());
                        const auto state = solver->tune(params, function, x0);

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
                                  << ",f=" << f0 << "/" << f << ",g=" << g << "[" << to_string(state.m_status) << "]"
                                  << ",calls=" << (function.fcalls() - fcalls0)
                                  << "/" << (function.gcalls() - gcalls0) << ".\n";

                        // check function value decrease
                        NANO_CHECK_LESS_EQUAL(f, f0 + epsilon1<scalar_t>());

                        // check convergence
                        NANO_CHECK_LESS_EQUAL(g, epsilon2<scalar_t>());
                        NANO_CHECK_EQUAL(state.m_status, opt_status::converged);
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
