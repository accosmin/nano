#pragma once

#include <limits>
#include <eigen3/Eigen/Core>

namespace ncv
{
        namespace min
        {
                ///
                /// \brief optimization state described as:
                ///     current point (x),
                ///     function value (f),
                ///     gradient (g),
                ///     descent direction (d) &
                ///     line-search step (t)
                ///
                template
                <
                        typename tscalar_,
                        typename tsize
                >
                struct state_t
                {
                        typedef tscalar_                tscalar;
                        typedef Eigen::Matrix
                        <
                                tscalar,
                                Eigen::Dynamic,
                                1,
                                Eigen::ColMajor
                        >                               tvector;

                        ///
                        /// \brief constructor
                        ///
                        explicit state_t(tsize size = 0)
                                :       x(size), g(size), d(size),
                                        f(std::numeric_limits<tscalar>::max()),
                                        m_iterations(0),
                                        m_fcalls(0),
                                        m_gcalls(0)
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
                        /// \brief update current state (move t along the chosen direction)
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
                                m_fcalls = problem.fcalls();
                                m_gcalls = problem.gcalls();
                        }

                        ///
                        /// \brief update current state (move t along the chosen direction)
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
                                m_fcalls = problem.fcalls();
                                m_gcalls = problem.gcalls();
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
                        tsize iterations() const { return m_iterations; }
                        tsize fcalls() const { return m_fcalls; }
                        tsize gcalls() const { return m_gcalls; }

                        // attributes
                        tvector         x, g, d;                ///< parameter, gradient, descent direction
                        tscalar         f;                      ///< function value, step size
                        tsize           m_iterations;
                        tsize           m_fcalls;               ///< #function value evaluations
                        tsize           m_gcalls;               ///< #function gradient evaluations
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

