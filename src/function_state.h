#pragma once

#include "arch.h"
#include "tensor.h"
#include "text/cast.h"

namespace nano
{
        class function_state_t;
        using ref_function_state_t = std::reference_wrapper<const function_state_t>;

        class function_t;

        ///
        /// \brief compare two optimization states
        ///
        bool operator<(const function_state_t&, const function_state_t&);

        ///
        /// \brief create an optimization state at the given point
        ///
        function_state_t make_state(const function_t&, const vector_t& x);

        ///
        /// \brief create an optimization state at the given point, using the stochastic approximation
        ///
        function_state_t make_stoch_state(const function_t&, const vector_t& x);

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

        template <>
        inline enum_map_t<opt_status> enum_string<opt_status>()
        {
                return
                {
                        { opt_status::converged,   "converged" },
                        { opt_status::max_iters,   "max_iters" },
                        { opt_status::failed,      "failed" },
                        { opt_status::stopped,     "stopped" }
                };
        }
        ///
        /// \brief optimization state described as:
        ///     current point (x),
        ///     function value (f),
        ///     gradient (g),
        ///     descent direction (d) &
        ///     line-search step (t)
        ///
        class NANO_PUBLIC function_state_t
        {
        public:
                ///
                /// \brief constructor
                ///
                explicit function_state_t(const tensor_size_t size = 0);

                ///
                /// \brief update current state (move to another position)
                ///
                void update(const function_t&, const vector_t& xx);

                ///
                /// \brief update current state (move to another position) using the stochastic approximation
                ///
                void stoch_update(const function_t&, const vector_t& xx);

                ///
                /// \brief update current state (move t along the chosen direction)
                ///
                void update(const function_t&, const scalar_t t);

                ///
                /// \brief update current state (move t along the chosen direction) using the stochastic approximation
                ///
                void stoch_update(const function_t&, const scalar_t t);

                ///
                /// \brief update current state (move t along the chosen direction,
                /// but the function value & gradient are already computed)
                ///
                void update(const function_t&, const scalar_t t, const scalar_t ft, const vector_t& gt);

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

                ///
                /// \brief check divergence
                ///
                operator bool() const
                {
                        return std::isfinite(f) && std::isfinite(convergence_criteria());
                }

                // attributes
                vector_t        x, g, d;        ///< parameter, gradient, descent direction
                scalar_t        f;              ///< function value, step size
                opt_status      m_status;       ///< optimization status (todo: does it make sense to have it here?!)
        };

        ///
        /// \brief compare two optimization states
        ///
        inline bool operator<(const function_state_t& one, const function_state_t& two)
        {
                const auto f1 = std::isfinite(one.f) ? one.f : std::numeric_limits<scalar_t>::max();
                const auto f2 = std::isfinite(two.f) ? two.f : std::numeric_limits<scalar_t>::max();

                return f1 < f2;
        }
}
