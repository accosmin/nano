#include <utest/utest.h>
#include "solver.h"
#include "core/numeric.h"
#include "solvers/lsearch.h"

using namespace nano;

static void test(
        const solver_t& solver, const string_t& solver_id,
        const function_t& function, const vector_t& x0, const int iterations = 10000)
{
        const auto state0 = solver_state_t{function, x0};
        const auto f0 = state0.f;
        const auto g0 = state0.convergence_criteria();

        // minimize
        const auto state = solver.minimize(iterations, epsilon2<scalar_t>(), function, x0);
        const auto x = state.x;
        const auto f = state.f;
        const auto g = state.convergence_criteria();

        if (state.m_status != solver_state_t::status::converged)
        {
                std::cout << function.name() << " " << solver_id
                        << ": x = [" << x0.transpose() << "]/[" << x.transpose() << "]"
                        << ",f=" << f0 << "/" << f
                        << ",g=" << g0 << "/" << g
                        << "[" << to_string(state.m_status) << "]"
                        << ",calls=" << state.m_fcalls << "/" << state.m_gcalls << ".\n";

                json_t json;
                solver.to_json(json);
                std::cout << " ... using " << json.dump() << "\n";
        }

        // check function value decrease
        UTEST_CHECK_LESS_EQUAL(f, f0 + epsilon1<scalar_t>());

        // check convergence
        UTEST_CHECK_LESS(g, epsilon2<scalar_t>());
        UTEST_CHECK_EQUAL(state.m_status, solver_state_t::status::converged);
}

UTEST_BEGIN_MODULE(test_solvers)

UTEST_CASE(default_solvers)
{
        for (const auto& function : get_convex_functions(1, 4))
        {
                UTEST_REQUIRE(function);

                for (const auto& solver_id : get_solvers().ids(std::regex("!(dfp|nag|nagfr|broyden)")))
                {
                        const auto solver = get_solver(solver_id);
                        UTEST_REQUIRE(solver);

                        for (auto t = 0; t < 10; ++ t)
                        {
                                test(*solver, solver_id, *function, vector_t::Random(function->size()));
                        }
                }
        }
}

UTEST_CASE(lsearch_strategies)
{
        for (const auto& function : get_convex_functions(1, 4))
        {
                UTEST_REQUIRE(function);

                for (const auto& solver_id : {"gd", "cgd", "lbfgs", "bfgs"})
                {
                        const auto solver = get_solver(solver_id);
                        UTEST_REQUIRE(solver);

                        for (const auto lsearch_strategy : enum_values<lsearch_t::strategy>())
                        {
                                solver->from_json(to_json("strat", lsearch_strategy));

                                for (auto t = 0; t < 10; ++ t)
                                {
                                        test(*solver, solver_id, *function, vector_t::Random(function->size()));
                                }
                        }
                }
        }
}

UTEST_END_MODULE()
