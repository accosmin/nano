#include "function.h"
#include "solver_state.h"

using namespace nano;

solver_state_t::solver_state_t(const tensor_size_t size) :
        x(vector_t::Zero(size)),
        g(vector_t::Zero(size)),
        d(vector_t::Zero(size)),
        f(std::numeric_limits<scalar_t>::max()),
        m_status(opt_status::max_iters)
{
}

void solver_state_t::update(const function_t& function, const vector_t& xx)
{
        x = xx;
        f = function.eval(x, &g);
}

void solver_state_t::stoch_update(const function_t& function, const vector_t& xx)
{
        x = xx;
        f = function.stoch_eval(x, &g);
}

void solver_state_t::update(const function_t& function, const scalar_t t)
{
        x.noalias() += t * d;
        f = function.eval(x, &g);
}

void solver_state_t::stoch_update(const function_t& function, const scalar_t t)
{
        x.noalias() += t * d;
        f = function.stoch_eval(x, &g);
}

void solver_state_t::update(const function_t&, const scalar_t t, const scalar_t ft, const vector_t& gt)
{
        x.noalias() += t * d;
        f = ft;
        g = gt;
}

solver_state_t nano::make_state(const function_t& function, const vector_t& x)
{
        assert(function.size() == x.size());

        solver_state_t state(function.size());
        state.update(function, x);
        return state;
}

solver_state_t nano::make_stoch_state(const function_t& function, const vector_t& x)
{
        assert(function.size() == x.size());

        solver_state_t state(function.size());
        state.stoch_update(function, x);
        return state;
}
