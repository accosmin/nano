#ifndef NANOCV_OPTIMIZE_STATE_HPP
#define NANOCV_OPTIMIZE_STATE_HPP

#include "tensor/matrix.hpp"

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////
                // optimization state:
                //      current point (x), function value (f), gradient (g),
                //      descent direction (d) & line-search step (t).
                /////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tscalar_,
                        typename tsize
                >
                struct state_t
                {
                        typedef tscalar_                                                tscalar;
                        typedef typename tensor::vector_types_t<tscalar>::tvector       tvector;

                        // constructor
                        state_t(tsize size = 0)
                                :       x(size), g(size), d(size),
                                        f(std::numeric_limits<tscalar>::max()),
                                        t(1.0),
                                        m_iterations(0),
                                        m_n_fvals(0),
                                        m_n_grads(0)
                        {
                        }

                        // constructor
                        template
                        <
                                typename tproblem
                        >
                        state_t(const tproblem& problem, const tvector& x0)
                                :       state_t(problem.size()),
                                        x(x0)
                        {
                                f = problem(x, g);
                        }

                        // update current state
                        template
                        <
                                typename tproblem
                        >
                        void update(const tproblem& problem, tscalar t)
                        {
                                x.noalias() += t * d;
                                f = problem(x, g);

                                m_iterations ++;
                                m_n_fvals += problem.n_fval_calls();
                                m_n_grads += problem.n_grad_calls();
                        }

                        template
                        <
                                typename tproblem
                        >
                        void update(const tproblem& problem, tscalar t, tscalar ft, const tvector& gt)
                        {
                                x.noalias() += t * d;
                                f = ft;
                                g = gt;

                                m_iterations ++;
                                m_n_fvals = problem.n_fval_calls();
                                m_n_grads = problem.n_grad_calls();
                        }

                        // check convergence: the gradient is relatively small
                        bool converged(tscalar epsilon) const
                        {
                                return (g.template lpNorm<Eigen::Infinity>()) < epsilon * (1.0 + std::fabs(f));
                        }

                        // access functions
                        tsize n_iterations() const { return m_iterations; }
                        tsize n_fval_calls() const { return m_n_fvals; }
                        tsize n_grad_calls() const { return m_n_grads; }

                        // attributes
                        tvector         x, g, d;
                        tscalar         f, t;
                        tsize           m_iterations;
                        tsize           m_n_fvals;
                        tsize           m_n_grads;
                };

                // compare two optimization states
                template
                <
                        typename tscalar,
                        typename tsize
                >
                bool operator<(const state_t<tscalar, tsize>& one,
                               const state_t<tscalar, tsize>& other)
                {
                        return one.f < other.f;
                }
        }
}

#endif // NANOCV_OPTIMIZE_STATE_HPP
