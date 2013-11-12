#ifndef NANOCV_OPTIMIZE_RESULT_HPP
#define NANOCV_OPTIMIZE_RESULT_HPP

#include "core/optimize/state.hpp"

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////////
                // optimization result/solution.
                /////////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tscalar,
                        typename tsize
                >
                class result_t
                {
                public:

                        typedef state_t<tscalar, tsize>                 tstate;
                        typedef typename tstate::tvector                tvector;

                        // constructor (analytic gradient)
                        explicit result_t(tsize size = 0)
                                :       m_optimum(size),
                                        m_iterations(0),
                                        m_n_fvals(0),
                                        m_n_grads(0)
                        {
                        }

                        // update solution
                        template
                        <
                                typename tproblem
                        >
                        void update(const tproblem& problem, const tstate& state) { return _update(problem, state); }
                        void update(const result_t& result) { return _update(result); }

                        // access functions
                        const tstate& optimum() const { return m_optimum; }
                        tsize iterations() const { return m_iterations; }
                        tsize n_fval_calls() const { return m_n_fvals; }
                        tsize n_grad_calls() const { return m_n_grads; }

                private:

                        // implementation: update solution
                        template
                        <
                                typename tproblem
                        >
                        void _update(const tproblem& problem, const tstate& st)
                        {
                                m_n_fvals = problem.n_fval_calls();
                                m_n_grads = problem.n_grad_calls();

                                m_iterations ++;
                                if (st < m_optimum)
                                {
                                        m_optimum = st;
                                }
                        }

                        // implementation: update solution
                        void _update(const result_t& result)
                        {
                                m_n_fvals += result.m_n_fvals;
                                m_n_grads += result.m_n_grads;

                                m_iterations += result.m_iterations;
                                if (result.m_optimum < m_optimum)
                                {
                                        m_optimum = result.m_optimum;
                                }
                        }

                private:

                        // attributes
                        tstate                  m_optimum;              // optimum state
                        tsize                   m_iterations;           // #iterations
                        tsize                   m_n_fvals;              // #function value evaluations
                        tsize                   m_n_grads;              // #function gradient evaluations
                };
        }
}

#endif // NANOCV_OPTIMIZE_RESULT_HPP
