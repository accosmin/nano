#include "lsearch_init.h"

using namespace nano;

scalar_t lsearch_unit_init_t::get(const solver_state_t&)
{
        return 1;
}

scalar_t lsearch_linear_init_t::get(const solver_state_t& state)
{
        scalar_t t0;

        const auto dg = state.d.dot(state.g);
        switch (m_iteration ++)
        {
        case 0:
                t0 = 1;
                break;

        default:
                // NB: the line-search length is from the previous iteration!
                t0 = state.t * m_prevdg / dg;
                break;
        }

        m_prevdg = dg;
        return t0;
}

scalar_t lsearch_quadratic_init_t::get(const solver_state_t& state)
{
        scalar_t t0;

        switch (m_iteration ++)
        {
        case 0:
                t0 = 1;
                break;

        default:
                t0 = std::min(
                        scalar_t(1),
                        scalar_t(1.01) * 2 * (state.f - m_prevf) / state.d.dot(state.g));
                break;
        }

        m_prevf = state.f;
        return t0;
}

scalar_t lsearch_cgdescent_init_t::get(const solver_state_t& state)
{
        const auto phi0 = scalar_t(0.01);
        const auto xnorm = state.x.lpNorm<Eigen::Infinity>();
        const auto fnorm = std::fabs(state.f);

        if (xnorm > 0)
        {
                return phi0 * xnorm / state.g.lpNorm<Eigen::Infinity>();
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
