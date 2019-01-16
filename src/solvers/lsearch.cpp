#include "lsearch.h"
#include "lsearch_init.h"
#include "core/numeric.h"
#include "lsearch_backtrack.h"
#include "lsearch_cgdescent.h"
#include "lsearch_morethuente.h"

using namespace nano;

static std::unique_ptr<lsearch_init_t> make_initializer(const lsearch_t::initializer initializer)
{
        switch (initializer)
        {
        case lsearch_t::initializer::unit:              return std::make_unique<lsearch_unit_init_t>();
        case lsearch_t::initializer::linear:            return std::make_unique<lsearch_linear_init_t>();
        case lsearch_t::initializer::quadratic:         return std::make_unique<lsearch_quadratic_init_t>();
        case lsearch_t::initializer::cg_descent:        return std::make_unique<lsearch_cgdescent_init_t>();
        default:                                        assert(false); return nullptr;
        }
}

static std::unique_ptr<lsearch_strategy_t> make_strategy(const lsearch_t::strategy strategy)
{
        switch (strategy)
        {
        case lsearch_t::strategy::cg_descent:           return std::make_unique<lsearch_cgdescent_t>();
        case lsearch_t::strategy::more_thuente:         return std::make_unique<lsearch_morethuente_t>();
        case lsearch_t::strategy::backtrack_wolfe:      return std::make_unique<lsearch_backtrack_wolfe_t>();
        case lsearch_t::strategy::backtrack_armijo:     return std::make_unique<lsearch_backtrack_armijo_t>();
        case lsearch_t::strategy::backtrack_swolfe:     return std::make_unique<lsearch_backtrack_swolfe_t>();
        default:                                        assert(false); return nullptr;
        }
}

lsearch_t::lsearch_t(const initializer init, const strategy strat, const scalar_t c1, const scalar_t c2) :
        m_initializer(make_initializer(init)),
        m_strategy(make_strategy(strat))
{
        m_strategy->c1(c1);
        m_strategy->c2(c2);
        m_strategy->max_iterations(40);
}

bool lsearch_t::operator()(solver_state_t& state)
{
        // check descent direction
        if (!state.has_descent())
        {
                return false;
        }

        // check parameters
        if (!(  0 < m_strategy->c1() &&
                m_strategy->c1() < scalar_t(0.5) &&
                m_strategy->c1() < m_strategy->c2() &&
                m_strategy->c2() < 1))
        {
                return false;
        }

        // initial step length
        const auto t0 = nano::clamp(m_initializer->get(state), lsearch_strategy_t::stpmin(), scalar_t(1));

        // line-search step length
        const auto state0 = state;
        return m_strategy->get(state0, t0, state) && state && state < state0;
}
