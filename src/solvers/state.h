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
        ///     descent direction (d),
        ///     line-search step (t).
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
                template <typename tvector>
                void update(const function_t& function, const tvector& xx, const scalar_t tt = 0)
                {
                        t = tt;
                        x = xx;
                        f = function.vgrad(x, &g);
                }

                ///
                /// \brief update current state (move t along the chosen direction,
                /// but the function value & gradient are already computed)
                ///
                template <typename tvector>
                void update(const function_t&, const scalar_t tt, const scalar_t ft, const tvector& gt)
                {
                        t = tt;
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
                        return  std::isfinite(t) &&
                                std::isfinite(f) &&
                                std::isfinite(convergence_criteria());
                }

                ///
                /// \brief compute the dot product between the gradient and the descent direction
                ///
                auto dg() const
                {
                        return g.dot(d);
                }

                ///
                /// \brief check if the chosen direction is a descent direction
                ///
                auto has_descent() const
                {
                        return dg() < 0;
                }

                ///
                /// \brief check if the current step satisfies the Armijo condition (sufficient decrease)
                ///
                bool has_armijo(const solver_state_t& state0, const scalar_t c1) const
                {
                        assert(c1 > 0 && c1 < 1);
                        return f <= state0.f + t * c1 * state0.dg();
                }

                ///
                /// \brief check if the current step satisfies the Wolfe condition (sufficient curvature)
                ///
                bool has_wolfe(const solver_state_t& state0, const scalar_t c2) const
                {
                        assert(c2 > 0 && c2 < 1);
                        return dg() >= c2 * state0.dg();
                }

                ///
                /// \brief check if the current step satisfies the strong Wolfe condition (sufficient curvature)
                ///
                bool has_strong_wolfe(const solver_state_t& state0, const scalar_t c2) const
                {
                        assert(c2 > 0 && c2 < 1);
                        return std::fabs(dg()) <= c2 * std::fabs(state0.dg());
                }

                ///
                /// \brief check if the current step satisfies the approximate Wolfe condition (sufficient curvature)
                ///     see CG_DESCENT
                ///
                bool has_approx_wolfe(const solver_state_t& state0, const scalar_t c1, const scalar_t c2,
                        const scalar_t epsilon) const
                {
                        assert(0 < c1 && c1 < c2 && c2 < 1);
                        return  (2 * c1 - 1) * state0.dg() >= dg() &&
                                dg() >= +c2 * state0.dg() &&
                                f <= state0.f + epsilon;
                }

                // attributes
                vector_t        x, g, d;                        ///< parameter, gradient, descent direction
                scalar_t        f{0}, t{1};                     ///< function value, step size
                status          m_status{status::max_iters};    ///< optimization status
                size_t          m_fcalls{0};                    ///< #function value evaluations so far
                size_t          m_gcalls{0};                    ///< #function gradient evaluations so far
                size_t          m_iterations{0};                ///< #optimization iterations so far
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
