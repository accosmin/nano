#pragma once

#include <limits>
#include "arch.h"
#include "tensor.h"
#include "text/enum_string.hpp"

namespace nano
{
        struct state_t;
        class problem_t;

        bool operator<(const state_t& one, const state_t& two);

        ///
        /// \brief optimization status
        ///
        enum class opt_status
        {
                converged,      ///< convergence criteria reached
                max_iters,      ///< maximum number of iterations reached without convergence (default)
                failed,         ///< optimization failed (e.g. line-search failed)
                stopped         ///< user requested stop
        };

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
                state_t(const problem_t& problem, const vector_t& x0);

                ///
                /// \brief update current state (move to another position)
                ///
                void update(const problem_t& problem, const vector_t& xx);

                ///
                /// \brief update current state (move t along the chosen direction)
                ///
                void update(const problem_t& problem, const scalar_t t);

                ///
                /// \brief update current state (move t along the chosen direction,
                /// but the function value & gradient are already computed)
                ///
                void update(const problem_t& problem, const scalar_t t, const scalar_t ft, const vector_t& gt);

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
                        return (g.lpNorm<Eigen::Infinity>()) / (1 + std::fabs(f));
                }

                // attributes
                vector_t        x, g, d;                ///< parameter, gradient, descent direction
                scalar_t        f;                      ///< function value, step size

                std::size_t     m_iterations;
                std::size_t     m_fcalls;               ///< #function value evaluations
                std::size_t     m_gcalls;               ///< #function gradient evaluations
                opt_status      m_status;               ///<
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

        template <>
        inline std::map<opt_status, string_t> enum_string<opt_status>()
        {
                return
                {
                        { opt_status::converged,   "converged" },
                        { opt_status::max_iters,   "max_iters" },
                        { opt_status::failed,      "failed" },
                        { opt_status::stopped,     "stopped" }
                };
        }


}

