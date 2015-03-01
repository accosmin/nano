#pragma once

#include <algorithm>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief initial step length using a local quadratic interpolation
                ///     (Nocedal & Wright (numerical optimization 2nd) @ p.59)
                ///
                /// \note suitable for GD or CGD methods
                ///
                template
                <
                        typename tstate,

                        typename tscalar = typename tstate::tscalar
                >
                class linesearch_init_interpolation
                {
                public:

                        linesearch_init_interpolation()
                                :       m_first(true),
                                        m_prevf(0)
                        {
                        }

                        tscalar update(const tstate& cstate)
                        {
                                const tscalar dg = cstate.d.dot(cstate.g);

                                const tscalar t0 = m_first ?
                                        tscalar(1.0) :
                                        std::min(tscalar(1.0), tscalar(1.01 * 2.0 * (cstate.f - m_prevf) / dg));

                                m_first = false;
                                m_prevf = cstate.f;

                                return t0;
                        }

                private:

                        bool            m_first;        ///< check if first iteration
                        tscalar         m_prevf;        ///< previous function evaluation
                };
        }
}

