#include "lsearch_init.h"

using namespace nano;

scalar_t lsearch_unit_init_t::get(const solver_state_t& state)
{
        scalar_t t0 = 1;

        if (m_first)
        {
                // following CG_DESCENT's initial procedure ...
                const auto phi0 = scalar_t(0.01);
                const auto xnorm = state.x.lpNorm<Eigen::Infinity>();
                const auto fnorm = std::fabs(state.f);

                if (xnorm > 0)
                {
                        t0 = phi0 * xnorm / state.g.template lpNorm<Eigen::Infinity>();
                }
                else if (fnorm > 0)
                {
                        t0 = phi0 * fnorm / state.g.squaredNorm();
                }
                else
                {
                        t0 = 1;
                }
        }

        else
        {
                t0 = 1;
        }

        // OK, keep track of previous function value & step length
        m_first = false;
        m_prevf = state.f;
        m_prevt0 = t0;
        return t0;
}

scalar_t lsearch_quadratic_init_t::get(const solver_state_t& state)
{
        scalar_t t0 = 1;

        if (m_first)
        {
                // following CG_DESCENT's initial procedure ...
                const auto phi0 = scalar_t(0.01);
                const auto xnorm = state.x.lpNorm<Eigen::Infinity>();
                const auto fnorm = std::fabs(state.f);

                if (xnorm > 0)
                {
                        t0 = phi0 * xnorm / state.g.template lpNorm<Eigen::Infinity>();
                }
                else if (fnorm > 0)
                {
                        t0 = phi0 * fnorm / state.g.squaredNorm();
                }
                else
                {
                        t0 = 1;
                }
        }

        else
        {
                const auto dg = state.d.dot(state.g);
                const auto ro = scalar_t(1.01 * 2.0);

                t0 = std::min(scalar_t(1), ro * (state.f - m_prevf) / dg);
        }

        // OK, keep track of previous function value & step length
        m_first = false;
        m_prevf = state.f;
        m_prevt0 = t0;
        return t0;
}

scalar_t lsearch_consistent_init_t::get(const solver_state_t& state)
{
        scalar_t t0 = 1;

        if (m_first)
        {
                // following CG_DESCENT's initial procedure ...
                const auto phi0 = scalar_t(0.01);
                const auto xnorm = state.x.lpNorm<Eigen::Infinity>();
                const auto fnorm = std::fabs(state.f);

                if (xnorm > 0)
                {
                        t0 = phi0 * xnorm / state.g.template lpNorm<Eigen::Infinity>();
                }
                else if (fnorm > 0)
                {
                        t0 = phi0 * fnorm / state.g.squaredNorm();
                }
                else
                {
                        t0 = 1;
                }

                // also, keep track of previous direction
                m_prevdg = state.d.dot(state.g);
        }

        else
        {
                const auto dg = state.d.dot(state.g);
                t0 = (m_prevt0 * m_prevdg / dg);
                m_prevdg = dg;
        }

        // OK, keep track of previous function value & step length
        m_first = false;
        m_prevf = state.f;
        m_prevt0 = t0;
        return t0;
}
