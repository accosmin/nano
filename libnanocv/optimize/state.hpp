#pragma once

#include <eigen3/Eigen/Core>
#include <limits>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief optimization state described as:
                /// current point (x),
                /// function value (f),
                /// gradient (g),
                /// descent direction (d) &
                /// line-search step (t)
                ///
                template
                <
                        typename tscalar_,
                        typename tsize
                >
                struct state_t
                {
                        typedef tscalar_                                                tscalar;
                        typedef Eigen::Matrix
                        <
                                tscalar,
                                Eigen::Dynamic,
                                1,
                                Eigen::ColMajor
                        >                                                               tvector;

                        ///
                        /// \brief constructor
                        ///
                        explicit state_t(tsize size = 0)
                                :       x(size), g(size), d(size),
                                        f(std::numeric_limits<tscalar>::max()),
                                        m_iterations(0),
                                        m_n_fvals(0),
                                        m_n_grads(0)
                        {
                        }

                        ///
                        /// \brief constructor
                        ///
                        template
                        <
                                typename tproblem
                        >
                        state_t(const tproblem& problem, const tvector& x0)
                                :       state_t(problem.size())
                        {
                                x = x0;
                                f = problem(x, g);
                        }

                        ///
                        /// \brief update current state
                        ///
                        template
                        <
                                typename tproblem
                        >
                        void update(const tproblem& problem, tscalar t)
                        {
                                x.noalias() += t * d;
                                f = problem(x, g);

                                m_iterations ++;
                                m_n_fvals = problem.n_fval_calls();
                                m_n_grads = problem.n_grad_calls();
                        }

                        ///
                        /// \brief update current state
                        ///
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

                        ///
                        /// \brief check convergence: the gradient is relatively small
                        ///
                        bool converged(tscalar epsilon) const
                        {
                                return convergence_criteria() < epsilon;
                        }

                        ///
                        /// \brief convergence criteria: relative gradient
                        ///
                        tscalar convergence_criteria() const
                        {
                                return (g.template lpNorm<Eigen::Infinity>()) / (1.0 + std::fabs(f));
                        }

                        // access functions
                        tsize n_iterations() const { return m_iterations; }
                        tsize n_fval_calls() const { return m_n_fvals; }
                        tsize n_grad_calls() const { return m_n_grads; }

                        // attributes
                        tvector         x, g, d;                ///< parameter, gradient, descent direction
                        tscalar         f;                      ///< function value, step size
                        tsize           m_iterations;
                        tsize           m_n_fvals;
                        tsize           m_n_grads;
                };

                ///
                /// \brief compare two optimization states
                ///
                template
                <
                        typename tscalar,
                        typename tsize
                >
                bool operator<(const state_t<tscalar, tsize>& one,
                               const state_t<tscalar, tsize>& two)
                {
                        const tscalar f1 = std::isfinite(one.f) ? one.f : std::numeric_limits<tscalar>::max();
                        const tscalar f2 = std::isfinite(two.f) ? two.f : std::numeric_limits<tscalar>::max();

                        return f1 < f2;
                }
        }
}

