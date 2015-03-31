#pragma once

#include "linesearch.h"
#include <algorithm>

namespace ncv
{
        namespace optimize
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
                class linesearch_init_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        explicit linesearch_init_t(ls_initializer type)
                                :       m_type(type),
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

                                switch (m_type)
                                {
                                case ls_initializer::consistent:
                                        {
                                                const tscalar dg = cstate.d.dot(cstate.g);

                                                t0 =    m_first ? unit :
                                                        (m_prevt0 * m_prevdg / dg);

                                                m_prevdg = dg;
                                                m_prevt0 = t0;
                                        }
                                        break;

                                case ls_initializer::quadratic:
                                        {
                                                const tscalar dg = cstate.d.dot(cstate.g);

                                                t0 =    m_first ? unit :
                                                        std::min(unit, tscalar(1.01 * 2.0 * (cstate.f - m_prevf) / dg));

                                                m_prevf = cstate.f;
                                        }
                                        break;

                                case ls_initializer::unit:
                                default:
                                        break;
                                }

                                // OK
                                m_first = false;
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
}

