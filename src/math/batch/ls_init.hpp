#pragma once

#include "math/lsearch_types.h"
#include <algorithm>

namespace nano
{
        ///
        /// \brief heuristics to initialize the step length,
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
        ///
        template
        <
                typename tstate,
                typename tscalar = typename tstate::tscalar
        >
        class ls_init_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit ls_init_t(ls_initializer type) :
                        m_type(type),
                        m_first(true),
                        m_prevf(0),
                        m_prevt0(1),
                        m_prevdg(1)
                {
                }

                ///
                /// \brief compute the initial step length
                ///
                tscalar operator()(const tstate& cstate)
                {
                        const tscalar unit = tscalar(1.0);

                        tscalar t0 = unit;

                        if (m_first)
                        {
                                // following CG_DESCENT's initial procedure ...
                                const tscalar phi0(0.01);

                                const tscalar xnorm = cstate.x.template lpNorm<Eigen::Infinity>();
                                const tscalar fnorm = std::fabs(cstate.f);

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
                                                const tscalar dg = cstate.d.dot(cstate.g);

                                                t0 = (m_prevt0 * m_prevdg / dg);

                                                m_prevdg = dg;
                                        }
                                        break;

                                case ls_initializer::quadratic:
                                        {
                                                const tscalar dg = cstate.d.dot(cstate.g);
                                                const tscalar ro = tscalar(1.01 * 2.0);

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

        private:

                ls_initializer  m_type;
                bool            m_first;        ///< check if first iteration
                tscalar         m_prevf;        ///< previous function evaluation
                tscalar         m_prevt0;       ///< previous step length
                tscalar         m_prevdg;       ///< previous direction dot product
        };
}

