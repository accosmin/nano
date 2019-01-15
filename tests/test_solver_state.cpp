#include <utest/utest.h>
#include "solvers/state.h"
#include "functions/sphere.h"

using namespace nano;

// todo: check convergence criterion
// todo: check armijo condition
// todo: check wolfe conditions
// todo: check update
// todo: check approximative wolfe conditions (cg_descent)

UTEST_BEGIN_MODULE(test_solver_state)

UTEST_CASE(state_valid1)
{
        solver_state_t state(11);
        UTEST_CHECK(state);
}

UTEST_CASE(state_valid2)
{
        function_sphere_t function(7);

        solver_state_t state(function, vector_t::Random(function.size()));
        UTEST_CHECK(state);
}

UTEST_CASE(state_invalid1)
{
        solver_state_t state(11);
        state.t = INFINITY;
        UTEST_CHECK(!state);
}

UTEST_CASE(state_invalid2)
{
        solver_state_t state(11);
        state.f = NAN;
        UTEST_CHECK(!state);
}

UTEST_CASE(state_has_descent)
{
        function_sphere_t function(7);

        solver_state_t state(function, vector_t::Random(function.size()));
        state.d = -state.g;
        UTEST_CHECK(state.has_descent());
}

UTEST_CASE(state_has_no_descent1)
{
        function_sphere_t function(7);

        solver_state_t state(function, vector_t::Random(function.size()));
        state.d.setZero();
        UTEST_CHECK(!state.has_descent());
}

UTEST_CASE(state_has_no_descent1)
{
        function_sphere_t function(7);

        solver_state_t state(function, vector_t::Random(function.size()));
        state.d = state.g;
        UTEST_CHECK(!state.has_descent());
}

UTEST_CASE(state_convergence)
{
}

UTEST_CASE(state_armijo)
{
}

UTEST_CASE(state_wolfe)
{
}

UTEST_CASE(state_approx_wolfe)
{
}

UTEST_END_MODULE()
