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

        switch (m_iteration ++)
        {
        case 0:
                t0 = first_step_length(state);
                break;

        default:
                t0 = 1;
                break;
        }

        return t0;
}

scalar_t lsearch_quadratic_init_t::get(const solver_state_t& state)
{
        scalar_t t0 = 1;

        switch (m_iteration ++)
        {
        case 0:
                t0 = first_step_length(state);
                break;

        default:
                t0 = std::min(scalar_t(1), scalar_t(1.01) * 2 * (state.f - m_prevf) / state.d.dot(state.g));
                break;
        }

        m_prevf = state.f;
        return t0;
}

scalar_t lsearch_consistent_init_t::get(const solver_state_t& state)
{
        scalar_t t0 = 1;

        const auto dg = state.d.dot(state.g);

        switch (m_iteration ++)
        {
        case 0:
                t0 = first_step_length(state);
                break;

        case 1:
                t0 = state.t * m_prevdg / dg; // NB: the line-search step is from the previous step!
                break;
        }

        m_prevdg = dg;
        return t0;
}
