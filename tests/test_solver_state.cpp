#include <utest/utest.h>
#include "core/numeric.h"
#include "solvers/state.h"
#include "functions/sphere.h"

using namespace nano;

// todo: check armijo condition
// todo: check wolfe conditions
// todo: check update
// todo: check approximative wolfe conditions (cg_descent)

static const function_sphere_t function(7);

UTEST_BEGIN_MODULE(test_solver_state)

UTEST_CASE(state_valid1)
{
        solver_state_t state(11);
        UTEST_CHECK(state);
}

UTEST_CASE(state_valid2)
{
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
        solver_state_t state(function, vector_t::Random(function.size()));
        state.d = -state.g;
        UTEST_CHECK(state.has_descent());
}

UTEST_CASE(state_has_no_descent0)
{
        solver_state_t state(function, vector_t::Random(function.size()));
        state.d.setZero();
        UTEST_CHECK(!state.has_descent());
}

UTEST_CASE(state_has_no_descent1)
{
        solver_state_t state(function, vector_t::Random(function.size()));
        state.d = state.g;
        UTEST_CHECK(!state.has_descent());
}

UTEST_CASE(state_convergence0)
{
        solver_state_t state(function, vector_t::Zero(function.size()));
        UTEST_CHECK(state.converged(epsilon2<scalar_t>()));
        UTEST_CHECK_GREATER_EQUAL(state.convergence_criterion(), 0);
        UTEST_CHECK_LESS(state.convergence_criterion(), epsilon0<scalar_t>());
}

UTEST_CASE(state_convergence1)
{
        solver_state_t state(function, vector_t::Random(function.size()) * epsilon1<scalar_t>());
        UTEST_CHECK(state.converged(epsilon2<scalar_t>()));
        UTEST_CHECK_GREATER_EQUAL(state.convergence_criterion(), 0);
        UTEST_CHECK_LESS(state.convergence_criterion(), epsilon2<scalar_t>());
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
