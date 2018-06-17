#include "lsearch.h"

using namespace nano;

static std::unique_ptr<lsearch_init_t> make_initializer(const lsearch_t::initializer initializer)
{
        switch (initializer)
        {
        case initializer::unit:                 return make_unique<lsearch_unit_init_t>();
        case initializer::quadratic:            return make_unique<lsearch_quadratic_init_t>();
        case initializer::consistent:           return make_unique<lsearch_consistent_init_t>();
        default:                                assert(false); return nullptr;
        }
}

static std::unique_ptr<lsearch_length_t> make_strategy(const lsearch_t::strategy strategy, const scalar_t c1, const scalar_t c2)
{
        assert(m_c1 < m_c2);
        assert(m_c1 > scalar_t(0) && m_c1 < scalar_t(1));
        assert(m_c2 > scalar_t(0) && m_c2 < scalar_t(1));

        switch (strategy)
        {
        case strategy::cg_descent:              return make_unique<lsearch_cgdescent_length_t>(c1, c2);
        case strategy::interpolation:           return make_unique<lsearch_interpolation_length_t>(c1, c2);
        case strategy::backtrack_wolfe:         return make_unique<lsearch_backtrack_wolfe_length_t>(c1, c2);
        case strategy::backtrack_armijo:        return make_unique<lsearch_backtrack_armijo_length_t>(c1, c2);
        case strategy::backtrack_strong_wolfe:  return make_unique<lsearch_backtrack_strong_wolfe_length_t>(c1, c2);
        default:                                assert(false); return nullptr;
        }
}

lsearch_t::lsearch_t(const initializer init, const strategy strat, const scalar_t c1, const scalar_t c2) :
        m_initializer(make_initializer(init)),
        m_strategy(make_strategy(strat, c1, c2))
{
}

bool lsearch_t::operator()(const function_t& function, solver_state_t& state)
{
        // initial step length
        const auto t0 = m_initializer->get(state);

        // check descent direction
        const scalar_t dg0 = state.d.dot(state.g);
        if (dg0 >= scalar_t(0))
        {
                return false;
        }

        // check initial step length
        if (t0 < lsearch_step_t::minimum() || t0 > lsearch_step_t::maximum())
        {
                return false;
        }

        // starting point
        lsearch_step_t step0(function, state);
        if (!step0)
        {
                return false;
        }

        // line-search step length
        const auto step = m_strategy->get(step0, t0);
        if (step && step < step0)
        {
                state.update(function, step.alpha(), step.func(), step.grad());
                return true;
        }
        else
        {
                return false;
        }
}
