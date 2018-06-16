#include <utility>
#include "lsearch.h"
#include "math/cubic.h"
#include "math/quadratic.h"

using namespace nano;

///
/// \brief bisection interpolation in the [step0, step1] line-search interval
///
static auto ls_bisection(const lsearch_step_t& step0, const lsearch_step_t& step1)
{
        return (step0.alpha() + step1.alpha()) / 2;
}

///
/// \brief quadratic interpolation in the [step0, step1] line-search interval
///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.58
///
/// NB: using the gradient at step0
/// NB: the step-length at step0 may be different than zero
///
static auto ls_quadratic(const lsearch_step_t& step0, const lsearch_step_t& step1)
{
        const auto x0 = step0.alpha(), f0 = step0.phi(), g0 = step0.gphi();
        const auto x1 = step1.alpha(), f1 = step1.phi();

        auto min = std::numeric_limits<scalar_t>::infinity();

        const auto q = quadratic_t<scalar_t>{x0, f0, g0, x1, f1};
        if (q)
        {
                q.extremum(min);
        }

        return min;
}

///
/// \brief cubic interpolation in the [step0, step1] line-search interval
///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
///
static auto ls_cubic(const lsearch_step_t& step0, const lsearch_step_t& step1)
{
        const auto x0 = step0.alpha(), f0 = step0.phi(), g0 = step0.gphi();
        const auto x1 = step1.alpha(), f1 = step1.phi(), g1 = step1.gphi();

        auto min1 = std::numeric_limits<scalar_t>::infinity();
        auto min2 = std::numeric_limits<scalar_t>::infinity();

        const auto c = cubic_t<scalar_t>{x0, f0, g0, x1, f1, g1};
        if (c)
        {
                c.extremum(min1, min2);
        }

        return std::make_pair(min1, min2);
}

lsearch_t::lsearch_t(const initializer init, const strategy strat, const scalar_t c1, const scalar_t c2) :
        m_initializer(init),
        m_strategy(strat),
        m_first(true),
        m_prevf(0),
        m_prevt0(1),
        m_prevdg(1),
        m_c1(c1),
        m_c2(c2)
{
        assert(m_c1 < m_c2);
        assert(m_c1 > scalar_t(0) && m_c1 < scalar_t(1));
        assert(m_c2 > scalar_t(0) && m_c2 < scalar_t(1));
}

scalar_t lsearch_t::t0(const solver_state_t& cstate)
{
        scalar_t t0 = 1;

        if (m_first)
        {
                // following CG_DESCENT's initial procedure ...
                const auto phi0 = scalar_t(0.01);
                const auto xnorm = cstate.x.lpNorm<Eigen::Infinity>();
                const auto fnorm = std::fabs(cstate.f);

                if (xnorm > 0)
                {
                        t0 = phi0 * xnorm / cstate.g.template lpNorm<Eigen::Infinity>();
                }
                else if (fnorm > 0)
                {
                        t0 = phi0 * fnorm / cstate.g.squaredNorm();
                }
                else
                {
                        t0 = 1;
                }

                // also, keep track of previous direction
                switch (m_initializer)
                {
                case initializer::consistent:
                        m_prevdg = cstate.d.dot(cstate.g);
                        break;

                default:
                        break;
                }
        }

        else
        {
                switch (m_initializer)
                {
                case initializer::consistent:
                        {
                                const auto dg = cstate.d.dot(cstate.g);

                                t0 = (m_prevt0 * m_prevdg / dg);

                                m_prevdg = dg;
                        }
                        break;

                case initializer::quadratic:
                        {
                                const auto dg = cstate.d.dot(cstate.g);
                                const auto ro = scalar_t(1.01 * 2.0);

                                t0 = std::min(scalar_t(1), ro * (cstate.f - m_prevf) / dg);
                        }
                        break;

                case initializer::unit:
                        t0 = 1;
                        break;

                default:
                        assert(false);
                        break;
                }
        }

        // OK, keep track of previous function value & step length
        m_first = false;
        m_prevf = cstate.f;
        m_prevt0 = t0;
        return t0;
}

bool lsearch_t::operator()(const function_t& function, const scalar_t t0, solver_state_t& state)
{
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

        // line-search
        switch (m_strategy)
        {
        case ls_strategy::backtrack_armijo:
                return setup(function, step0, m_ls_backtrack_armijo(m_c1, m_c2, step0, t0), state);

        case ls_strategy::backtrack_wolfe:
                return setup(function, step0, m_ls_backtrack_wolfe(m_c1, m_c2, step0, t0), state);

        case ls_strategy::backtrack_strong_wolfe:
                return setup(function, step0, m_ls_backtrack_strong_wolfe(m_c1, m_c2, step0, t0), state);

        case ls_strategy::cg_descent:
                return setup(function, step0, m_ls_cgdescent(m_c1, m_c2, step0, t0), state);

        case ls_strategy::interpolation:
                return setup(function, step0, m_ls_interpolate(m_c1, m_c2, step0, t0), state);

        default:
                assert(false); return false;
        }
}

bool ls_strategy_t::setup(const function_t& function, const lsearch_step_t& step0, const lsearch_step_t& step, solver_state_t& state) const
{
        return step && step < step0 && setup(function, step, state);
}

bool ls_strategy_t::setup(const function_t& function, const lsearch_step_t& step, solver_state_t& state) const
{
        state.update(function, step.alpha(), step.func(), step.grad());
        return true;
}
