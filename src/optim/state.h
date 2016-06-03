#pragma once

#include <limits>
#include "arch.h"
#include "tensor.h"

namespace nano
{
        struct state_t;

        bool operator<(const state_t& one, const state_t& two);

        ///
        /// \brief optimization state described as:
        ///     current point (x),
        ///     function value (f),
        ///     gradient (g),
        ///     descent direction (d) &
        ///     line-search step (t)
        ///
        struct NANO_PUBLIC state_t
        {
                ///
                /// \brief constructor
                ///
                explicit state_t(const tensor_size_t size = 0);

                ///
                /// \brief constructor
                ///
                template <typename tproblem>
                state_t(const tproblem& problem, const vector_t& x0) :
                        state_t(problem.size())
                {
                        x = x0;
                        f = problem(x, g);
                }

                ///
                /// \brief update current state (move to another position)
                ///
                template <typename tproblem>
                void update(const tproblem& problem, const vector_t& xx)
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
                template <typename tproblem>
                void update(const tproblem& problem, const scalar_t t)
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
                template <typename tproblem>
                void update(const tproblem& problem, const scalar_t t, const scalar_t ft, const vector_t& gt)
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
                bool update(const state_t& state);

                ///
                /// \brief check convergence: the gradient is relatively small
                ///
                bool converged(const scalar_t epsilon) const
                {
                        return convergence_criteria() < epsilon;
                }

                ///
                /// \brief convergence criteria: relative gradient
                ///
                scalar_t convergence_criteria() const
                {
                        return (g.template lpNorm<Eigen::Infinity>()) / (1 + std::fabs(f));
                }

                // attributes
                vector_t        x, g, d;                ///< parameter, gradient, descent direction
                scalar_t        f;                      ///< function value, step size

                std::size_t     m_iterations;
                std::size_t     m_fcalls;               ///< #function value evaluations
                std::size_t     m_gcalls;               ///< #function gradient evaluations
        };

        ///
        /// \brief compare two optimization states
        ///
        inline bool operator<(const state_t& one, const state_t& two)
        {
                const auto f1 = std::isfinite(one.f) ? one.f : std::numeric_limits<scalar_t>::max();
                const auto f2 = std::isfinite(two.f) ? two.f : std::numeric_limits<scalar_t>::max();

                return f1 < f2;
        }
}

