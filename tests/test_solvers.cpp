#include <utest/utest.h>
#include "solver.h"
#include "core/numeric.h"

using namespace nano;

UTEST_BEGIN_MODULE(test_solvers)

UTEST_CASE(evaluate)
{
        for (const auto& function : get_convex_functions(1, 4))
        {
                UTEST_REQUIRE(function);

                const auto iterations = size_t(1000);
                const auto trials = size_t(10);

                for (const auto& id : get_solvers().ids())
                {
                        const auto solver = get_solver(id);
                        UTEST_REQUIRE(solver);

                        if (    id == "dfp" ||
                                id == "nag" ||
                                id == "nagfr" ||
                                id == "broyden")
                        {
                                std::cout << "warning: skiping solver " << id << " (to fix)\n";
                                continue;
                        }

                        for (size_t t = 0; t < trials; ++ t)
                        {
                                const auto x0 = vector_t::Random(function->size());

                                const auto state0 = solver_state_t{*function, x0};
                                const auto f0 = state0.f;
                                const auto g0 = state0.convergence_criteria();

                                // optimize
                                const auto state = solver->minimize(iterations, epsilon2<scalar_t>(), *function, x0);
                                const auto x = state.x;
                                const auto f = state.f;
                                const auto g = state.convergence_criteria();

                                if (state.m_status != solver_state_t::status::converged)
                                {
                                        std::cout << function->name() << " " << id
                                                  << " [" << (t + 1) << "/" << trials << "]"
                                                  << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                                                  << ",f=" << f0 << "/" << f
                                                  << ",g=" << g0 << "/" << g
                                                  << "[" << to_string(state.m_status) << "]"
                                                  << ",calls=" << state.m_fcalls << "/" << state.m_gcalls << ".\n";
                                }

                                // check function value decrease
                                UTEST_CHECK_LESS_EQUAL(f, f0 + epsilon1<scalar_t>());

                                // check convergence
                                UTEST_CHECK_LESS(g, epsilon2<scalar_t>());
                                UTEST_CHECK_EQUAL(state.m_status, solver_state_t::status::converged);
                        }

                        std::cout << function->name() << " " << id << ".\n";
                }
        }
}

UTEST_END_MODULE()
