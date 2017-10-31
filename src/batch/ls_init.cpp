#include "ls_init.h"
#include <algorithm>

using namespace nano;

ls_init_t::ls_init_t(const ls_initializer type) :
        m_type(type),
        m_first(true),
        m_prevf(0),
        m_prevt0(1),
        m_prevdg(1)
{
}

scalar_t ls_init_t::operator()(const function_state_t& cstate)
{
        const auto unit = scalar_t(1.0);

        scalar_t t0 = unit;

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
                        t0 = unit;
                }

                // also, keep track of previous direction
                switch (m_type)
                {
                case ls_initializer::consistent:
                        m_prevdg = cstate.d.dot(cstate.g);
                        break;

                default:
                        break;
                }
        }

        else
        {
                switch (m_type)
                {
                case ls_initializer::consistent:
                        {
                                const auto dg = cstate.d.dot(cstate.g);

                                t0 = (m_prevt0 * m_prevdg / dg);

                                m_prevdg = dg;
                        }
                        break;

                case ls_initializer::quadratic:
                        {
                                const auto dg = cstate.d.dot(cstate.g);
                                const auto ro = scalar_t(1.01 * 2.0);

                                t0 = std::min(unit, ro * (cstate.f - m_prevf) / dg);
                        }
                        break;

                case ls_initializer::unit:
                default:
                        t0 = unit;
                        break;
                }
        }

        // OK, keep track of previous function value & step length
        m_first = false;
        m_prevf = cstate.f;
        m_prevt0 = t0;
        return t0;
}
