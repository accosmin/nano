#include "utest.h"
#include "solver.h"
#include "math/numeric.h"
#include "math/epsilon.h"

using namespace nano;

NANO_BEGIN_MODULE(test_solvers)

NANO_CASE(evaluate)
{
        for (const auto& function : get_convex_functions(1, 4))
        {
                NANO_REQUIRE(function);

                const auto iterations = size_t(1000);
                const auto trials = size_t(100);

                // solvers to try
                for (const auto& id : get_batch_solvers().ids())
                {
                        const auto solver = get_batch_solvers().get(id);
                        NANO_REQUIRE(solver);

                        size_t out_of_domain = 0;
                        for (size_t t = 0; t < trials; ++ t)
                        {
                                const auto x0 = vector_t::Random(function->size());
                                const auto f0 = function->eval(x0);
                                const auto fcalls0 = function->fcalls();
                                const auto gcalls0 = function->gcalls();

                                // optimize
                                const auto params = batch_params_t(iterations, epsilon2<scalar_t>());
                                const auto state = solver->minimize(params, *function, x0);

                                const auto x = state.x;
                                const auto f = state.f;
                                const auto g = state.convergence_criteria();

                                // ignore out-of-domain solutions
                                if (!function->is_valid(x))
                                {
                                        out_of_domain ++;
                                        continue;
                                }

                                std::cout << function->name() << " " << id
                                          << " [" << (t + 1) << "/" << trials << "]"
                                          << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                                          << ",f=" << f0 << "/" << f << ",g=" << g << "[" << to_string(state.m_status) << "]"
                                          << ",calls=" << (function->fcalls() - fcalls0)
                                          << "/" << (function->gcalls() - gcalls0) << ".\n";

                                // check function value decrease
                                NANO_CHECK_LESS_EQUAL(f, f0 + epsilon1<scalar_t>());

                                // check convergence
                                NANO_CHECK_LESS(g, epsilon2<scalar_t>());
                                NANO_CHECK_EQUAL(state.m_status, opt_status::converged);
                        }

                        std::cout << function->name() << " " << id
                                  << ": out of domain " << out_of_domain << "/" << trials << ".\n";
                }
        }
}

NANO_END_MODULE()

