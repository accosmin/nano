#include "lsearch_init.h"

using namespace nano;

static scalar_t first_step_length(const solver_state_t& state)
{
        // following CG_DESCENT's initial procedure ...
        const auto phi0 = scalar_t(0.01);
        const auto xnorm = state.x.lpNorm<Eigen::Infinity>();
        const auto fnorm = std::fabs(state.f);

        if (xnorm > 0)
        {
                return phi0 * xnorm / state.g.template lpNorm<Eigen::Infinity>();
        }
        else if (fnorm > 0)
        {
                return phi0 * fnorm / state.g.squaredNorm();
        }
        else
        {
                return 1;
        }
}

scalar_t lsearch_unit_init_t::get(const solver_state_t& state)
{
        scalar_t t0 = 1;

        if (m_first)
        {
                m_first = false;
                t0 = first_step_length(state);
        }

        else
        {
                t0 = 1;
        }

        return t0;
}

scalar_t lsearch_quadratic_init_t::get(const solver_state_t& state)
{
        scalar_t t0 = 1;

        if (m_first)
        {
                m_first = false;
                t0 = first_step_length(state);
        }

        else
        {
                const auto dg = state.d.dot(state.g);
                const auto ro = scalar_t(1.01 * 2.0);

                t0 = std::min(scalar_t(1), ro * (state.f - m_prevf) / dg);
        }

        m_prevf = state.f;
        return t0;
}

scalar_t lsearch_consistent_init_t::get(const solver_state_t& state)
{
        scalar_t t0 = 1;

        const auto dg = state.d.dot(state.g);
        if (m_first)
        {
                m_first = false;
                t0 = first_step_length(state);
        }

        else
        {
                t0 = m_prevt * m_prevdg / dg;
        }

        m_prevdg = dg;
        m_prevf = state.f;
        m_prevt = state.t;
        return t0;
}
