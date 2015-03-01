#pragma once

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief initial step length using the consistency of the first-order change of the function value
                ///     (Nocedal & Wright (numerical optimization 2nd) @ p.59)
                ///
                /// \note suitable for GD or CGD methods
                ///
                template
                <
                        typename tstate,

                        typename tscalar = typename tstate::tscalar
                >
                class linesearch_init_consistency
                {
                public:

                        linesearch_init_consistency()
                                :       m_first(true),
                                        m_prevt0(1),
                                        m_prevdg(1)
                        {
                        }

                        tscalar update(const tstate& cstate)
                        {
                                const tscalar dg = cstate.d.dot(cstate.g);

                                const tscalar t0 = m_first ?
                                        tscalar(1.0) :
                                        (m_prevt0 * m_prevdg / dg);

                                m_first = false;
                                m_prevdg = dg;
                                m_prevt0 = t0;

                                return t0;
                        }

                private:

                        bool            m_first;        ///< check if first iteration
                        tscalar         m_prevt0;       ///< previous step length
                        tscalar         m_prevdg;       ///< previous direction dot product
                };
        }
}

