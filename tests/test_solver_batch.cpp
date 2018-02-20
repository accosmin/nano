#include "utest.h"
#include "function.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "solver_batch.h"

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
        const auto iterations = size_t(1000);
        const auto trials = size_t(100);

        // solvers to try
        for (const auto& id : get_batch_solvers().ids())
        {
                const auto solver = get_batch_solvers().get(id);
                NANO_REQUIRE(solver);

                solver->config(json_writer_t().object("c1", epsilon0<scalar_t>()).str());

                size_t out_of_domain = 0;

                for (size_t t = 0; t < trials; ++ t)
                {
                        const auto x0 = vector_t::Random(function.size());
                        const auto f0 = function.eval(x0);

                        // optimize
                        const auto params = batch_params_t(iterations, epsilon2<scalar_t>());
                        const auto state = solver->minimize(params, function, x0);

                        const auto x = state.x;
                        const auto f = state.f;
                        const auto g = state.convergence_criteria();

                        // ignore out-of-domain solutions
                        if (!function.is_valid(x))
                        {
                                out_of_domain ++;
                                continue;
                        }

                        std::cout << function.name() << " " << id
                                  << " [" << (t + 1) << "/" << trials << "]"
                                  << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                                  << ",f=" << f0 << "/" << f << ",g=" << g << "[" << to_string(state.m_status) << "]"
                                  << ",calls=" << function.fcalls() << "/" << function.gcalls() << ".\n";

                        // check function value decrease
                        NANO_CHECK_LESS_EQUAL(f, f0 + epsilon1<scalar_t>());

                        // check convergence
                        NANO_CHECK_LESS(g, epsilon2<scalar_t>());
                        NANO_CHECK_EQUAL(state.m_status, opt_status::converged);
                }

                std::cout << function.name() << ", " << id
                          << ": out of domain " << out_of_domain << "/" << trials << ".\n";
        }
}

NANO_BEGIN_MODULE(test_batch_solvers)

NANO_CASE(evaluate)
{
        for (const auto& function : get_convex_functions(1, 4))
        {
                check_function(*function);
        }
}

NANO_END_MODULE()

