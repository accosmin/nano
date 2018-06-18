#include "lsearch_init_unit.h"

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
