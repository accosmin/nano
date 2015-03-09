#pragma once

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief line-search step
                ///
                template
                <
                        typename tscalar
                >
                struct linesearch_step_t
                {
                        linesearch_step_t(const tscalar step)
                                :       m_step(step),
                                        m_func(),
                                        m_grad(0)
                        {
                        }

                        template
                        <
                                typename tproblem,
                                typename tstate,
                                typename tvector
                        >
                        tscalar update(const tproblem& problem, const tstate& state, tvector& g)
                        {
                                m_func = problem(state.x + m_step * state.d, g);
                                m_grad = state.d.dot(g);
                                return m_func;
                        }

                        tscalar         m_step;         ///< step (>0)
                        tscalar         m_func;         ///< function value along the search direction
                        tscalar         m_grad;         ///< function gradient along the search direction
                };
        }
}

