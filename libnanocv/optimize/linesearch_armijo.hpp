#pragma once

#include <limits>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief line-search method to find the scalar that reduces
                /// the function value (the most) along the direction d: argmin(t) f(x + t * d),
                /// using the Armijo (sufficient decrease) condition.
                ///
                template
                <
                        typename tproblem,

                        // dependent types
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tstate = typename tproblem::tstate
                >
                class linesearch_armijo_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        linesearch_armijo_t(tscalar alpha = 0.2, tscalar beta = 0.7)
                                :       m_alpha(alpha),
                                        m_beta(beta)
                        {
                        }

                        ///
                        /// \brief update the current state
                        ///
                        bool update(const tproblem& problem, tscalar t0, tstate& state) const
                        {
                                const tscalar t = step(problem, t0, state);
                                if (t < std::numeric_limits<tscalar>::epsilon())
                                {
                                        // failed to find a suitable line-search step
                                        return false;
                                }
                                else
                                {
                                        // OK, update the current state
                                        state.update(problem, t);
                                        return true;
                                }
                        }

                private:

                        tscalar step(const tproblem& problem, tscalar t0, const tstate& state,
                                tsize max_iters = 64) const
                        {
                                const tscalar dg = state.d.dot(state.g);

                                // (Nocedal & Wright (numerical optimization 2nd) @ p.37)
                                tscalar t = t0;
                                for (tsize i = 0; i < max_iters; i ++, t = m_beta * t)
                                {
                                        if (problem(state.x + t * state.d) < state.f + t * m_alpha * dg)
                                        {
                                                return t;
                                        }
                                }

                                // OK, give up
                                return 0;
                        }

                private:

                        // attributes
                        tscalar         m_alpha;        ///< minimum function value's decrease rate
                        tscalar         m_beta;         ///< line-search step lenght's accuracy
                };
        }
}

