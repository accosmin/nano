#pragma once

#include "function.h"
#include "core/cast.h"

namespace nano
{
        class solver_state_t;
        using ref_solver_state_t = std::reference_wrapper<const solver_state_t>;

        ///
        /// \brief optimization state described as:
        ///     current point (x),
        ///     function value (f),
        ///     gradient (g),
        ///     descent direction (d) &
        ///     line-search step (t)
        ///
        class solver_state_t
        {
        public:

                enum class status
                {
                        converged,      ///< convergence criteria reached
                        max_iters,      ///< maximum number of iterations reached without convergence (default)
                        failed,         ///< optimization failed (e.g. line-search failed)
                        stopped         ///< user requested stop
                };

                solver_state_t() = default;

                ///
                /// \brief constructor
                ///
                explicit solver_state_t(const tensor_size_t size) :
                        x(vector_t::Zero(size)),
                        g(vector_t::Zero(size)),
                        d(vector_t::Zero(size)),
                        f(std::numeric_limits<scalar_t>::max())
                {
                }

                ///
                /// \brief constructor
                ///
                solver_state_t(const function_t& function, const vector_t& x0) :
                        solver_state_t(x0.size())
                {
                        assert(function.size() == x.size());
                        update(function, x0);
                }

                ///
                /// \brief update current state (move to another position)
                ///
                void update(const function_t& function, const vector_t& xx)
                {
                        x = xx;
                        f = function.vgrad(x, &g);
                }

                ///
                /// \brief update current state (move t along the chosen direction)
                ///
                void update(const function_t& function, const scalar_t t)
                {
                        x.noalias() += t * d;
                        f = function.vgrad(x, &g);
                }

                ///
                /// \brief update current state (move t along the chosen direction,
                /// but the function value & gradient are already computed)
                ///
                void update(const function_t&, const scalar_t t, const scalar_t ft, const vector_t& gt)
                {
                        x.noalias() += t * d;
                        f = ft;
                        g = gt;
                }

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
                        return g.lpNorm<Eigen::Infinity>() / std::max(scalar_t(1), std::fabs(f));
                }

                ///
                /// \brief check divergence
                ///
                operator bool() const
                {
                        return std::isfinite(f) && std::isfinite(convergence_criteria());
                }

                // attributes
                vector_t        x, g, d;                        ///< parameter, gradient, descent direction
                scalar_t        f{0};                           ///< function value, step size
                status          m_status{status::max_iters};    ///< optimization status
                size_t          m_fcalls{0};                    ///< #function value evaluations
                size_t          m_gcalls{0};                    ///< #function gradient evaluations
                size_t          m_iterations{0};                ///< #optimization iterations
        };

        template <>
        inline enum_map_t<solver_state_t::status> enum_string<solver_state_t::status>()
        {
                return
                {
                        { solver_state_t::status::converged,   "converged" },
                        { solver_state_t::status::max_iters,   "max_iters" },
                        { solver_state_t::status::failed,      "failed" },
                        { solver_state_t::status::stopped,     "stopped" }
                };
        }

        inline bool operator<(const solver_state_t& one, const solver_state_t& two)
        {
                return  (std::isfinite(one.f) ? one.f : std::numeric_limits<scalar_t>::max()) <
                        (std::isfinite(two.f) ? two.f : std::numeric_limits<scalar_t>::max());
        }

        inline std::ostream& operator<<(std::ostream& os, const solver_state_t::status status)
        {
                return os << to_string(status);
        }
}
