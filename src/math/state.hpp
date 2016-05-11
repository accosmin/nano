#pragma once

#include <limits>
#include <type_traits>
#include <eigen3/Eigen/Core>

namespace nano
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
                typename tvector_ = Eigen::Matrix<tscalar_, Eigen::Dynamic, 1, Eigen::ColMajor>,
                typename tsize_ = typename tvector_::Index,
                typename tvalid_tscalar = typename std::enable_if<std::is_floating_point<tscalar_>::value>::type
        >
        struct state_t
        {
                using tsize = tsize_;
                using tscalar = tscalar_;
                using tvector = tvector_;

                ///
                /// \brief constructor
                ///
                explicit state_t(const tsize size = 0) :
                        x(size), g(size), d(size),
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
                state_t(const tproblem& problem, const tvector& x0) :
                        state_t(problem.size())
                {
                        x = x0;
                        f = problem(x, g);
                }

                ///
                /// \brief update current state (move to another position)
                ///
                template
                <
                        typename tproblem
                >
                void update(const tproblem& problem, const tvector& xx)
                {
                        x = xx;
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
                void update(const tproblem& problem, tscalar t)
                {
                        x.noalias() += t * d;
                        f = problem(x, g);

                        m_iterations ++;
                        m_fcalls = problem.fcalls();
                        m_gcalls = problem.gcalls();
                }

                ///
                /// \brief update current state (move t along the chosen direction,
                /// but the function value & gradient are already computed)
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
                /// \brief update the current state (if an improvement)
                ///
                bool update(const state_t& state)
                {
                        const bool better = state < (*this);
                        if (better)
                        {
                                x = state.x;
                                g = state.g;
                                d = state.d;
                                f = state.f;
                        }

                        m_iterations = state.m_iterations;
                        m_fcalls = state.m_fcalls;
                        m_gcalls = state.m_gcalls;

                        return better;
                }

                ///
                /// \brief check convergence: the gradient is relatively small
                ///
                bool converged(const tscalar epsilon) const
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

                // attributes
                tvector         x, g, d;                ///< parameter, gradient, descent direction
                tscalar         f;                      ///< function value, step size

                std::size_t     m_iterations;
                std::size_t     m_fcalls;               ///< #function value evaluations
                std::size_t     m_gcalls;               ///< #function gradient evaluations
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

