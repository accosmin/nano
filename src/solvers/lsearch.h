#pragma once

#include "state.h"

namespace nano
{
        ///
        /// \brief compute the initial step length of the line search procedure.
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
        ///
        class lsearch_init_t
        {
        public:

                ///
                /// \brief constructor
                ///
                lsearch_init_t() = default;

                ///
                /// \brief destructor
                ///
                virtual ~lsearch_init_t() = default;

                ///
                /// \brief returns the initial step length given the current state
                /// NB: may keep track of the previous states
                ///
                scalar_t get(const solver_state_t& state)
                {
                        return get(state, m_iteration ++);
                }

        private:

                virtual scalar_t get(const solver_state_t&, const int iteration) = 0;

                // attributes
                int             m_iteration{0}; ///<
        };

        ///
        /// \brief compute the step length of the line search procedure.
        ///
        class lsearch_strategy_t
        {
        public:

                ///
                /// \brief constructor
                ///
                lsearch_strategy_t() = default;

                ///
                /// \brief destructor
                ///
                virtual ~lsearch_strategy_t() = default;

                ///
                /// \brief
                ///
                virtual bool get(const solver_state_t& state0, const scalar_t t0, solver_state_t&) = 0;

                ///
                /// \brief change parameters
                ///
                auto& c1(const scalar_t c1) { m_c1 = c1; return *this; }
                auto& c2(const scalar_t c2) { m_c2 = c2; return *this; }
                auto& max_iterations(const int max_iterations) { m_max_iterations = max_iterations; return *this; }

                ///
                /// \brief access functions
                ///
                auto c1() const { return m_c1; }
                auto c2() const { return m_c2; }
                auto max_iterations() const { return m_max_iterations; }

                ///
                /// \brief minimum allowed line-search step
                ///
                static scalar_t stpmin()
                {
                        return scalar_t(10) * std::numeric_limits<scalar_t>::epsilon();
                }

                ///
                /// \brief maximum allowed line-search step
                ///
                static scalar_t stpmax()
                {
                        return scalar_t(1) / stpmin();
                }

                ///
                /// \brief line-search step
                ///
                struct step_t
                {
                        step_t() = default;
                        step_t(const step_t&) = default;
                        step_t(const solver_state_t& state) : t(state.t), f(state.f), g(state.dg()) {}
                        step_t(const scalar_t tt, const scalar_t ff, const scalar_t gg) : t(tt), f(ff), g(gg) {}

                        step_t& operator=(const step_t&) = default;
                        step_t& operator=(const solver_state_t& state)
                        {
                                t = state.t, f = state.f, g = state.dg();
                                return *this;
                        }

                        scalar_t t{0};  ///< line-search step
                        scalar_t f{0};  ///< line-search function value
                        scalar_t g{0};  ///< line-search gradient
                };

                ///
                /// \brief cubic interpolation of two line-search steps.
                ///
                static auto cubic(const step_t& u, const step_t& v)
                {
                        // fit cubic: q(x) = a*x^3 + b*x^2 + c*x + d
                        //      given: q(u) = fu, q'(u) = gu
                        //      given: q(v) = fv, q'(v) = gv
                        // minimizer: solution of 3*a*x^2 + 2*b*x + c = 0
                        // see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
                        const auto d1 = u.g + v.g - 3 * (u.f - v.f) / (u.t - v.t);
                        const auto d2 = (v.t > u.t ? +1 : -1) * std::sqrt(d1 * d1 - u.g * v.g);
                        return v.t - (v.t - u.t) * (v.g + d2 - d1) / (v.g - u.g + 2 * d2);
                }

                ///
                /// \brief quadratic interpolation of two line-search steps.
                ///
                static auto quadratic(const step_t& u, const step_t& v, bool* convexity = nullptr)
                {
                        // fit quadratic: q(x) = a*x^2 + b*x + c
                        //      given: q(u) = fu, q'(u) = gu
                        //      given: q(v) = fv
                        // minimizer: -b/2a
                        const auto dt = u.t - v.t;
                        const auto df = u.f - v.f;
                        if (convexity)
                        {
                                *convexity = (u.g - df / dt) * dt > 0;
                        }
                        return u.t - u.g * dt * dt / (2 * (u.g * dt - df));
                }

                ///
                /// \brief secant interpolation of two line-search steps.
                ///
                static auto secant(const step_t& u, const step_t& v)
                {
                        // fit quadratic: q(x) = a*x^2 + b*x + c
                        //      given: q'(u) = gu
                        //      given: q'(v) = gv
                        // minimizer: -b/2a
                        return (v.t * u.g - u.t * v.g) / (u.g - v.g);
                }

                ///
                /// \brief bisection interpolation of two line-search steps.
                ///
                static auto bisection(const step_t& u, const step_t& v)
                {
                        // minimizer: (u+v)/2
                        return (u.t + v.t) / 2;
                }

                ///
                /// \brief interpolate two line-search steps
                ///     and choose the interpolation result closest to the minimum value point u.
                ///
                static auto interpolate(const step_t& u, const step_t& v)
                {
                        // NB: this doesn't assume the points u and v are sorted!
                        // NB: this assumes that u is the point with the minimum value!
                        const auto tc = cubic(u, v);
                        const auto tq = quadratic(u, v);
                        const auto ts = secant(u, v);
                        const auto tb = bisection(u, v);

                        scalar_t tt[4];
                        int tcount = 0;

                        const auto op_add = [&] (const auto t)
                        {
                                if (std::isfinite(t) && std::min(u.t, v.t) < t && t < std::max(u.t, v.t))
                                {
                                        tt[tcount ++] = t;
                                }
                        };

                        op_add(tc);
                        op_add(tq);
                        op_add(ts);
                        op_add(tb);

                        // choose the interpolation closest to the minimum u
                        const auto it = std::min_element(std::begin(tt), std::end(tt), [&] (const auto t1, const auto t2)
                        {
                                return std::fabs(t1 - u.t) < std::fabs(t2 - u.t);
                        });

                        return *it;
                }

        private:

                // attributes
                scalar_t        m_c1{static_cast<scalar_t>(1e-4)};      ///< sufficient decrease rate
                scalar_t        m_c2{static_cast<scalar_t>(0.1)};       ///< sufficient curvature
                int             m_max_iterations{40};                   ///< #maximum iterations
        };

        ///
        /// \brief line-search algorithm.
        ///
        class lsearch_t
        {
        public:

                ///
                /// \brief initial step length strategy
                ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
                ///     see CG_DESCENT
                ///
                enum class initializer
                {
                        unit,                           ///< 1.0 (useful for quasi-Newton and Newton methods)
                        linear,                         ///< consistent first-order change in the function
                        quadratic,                      ///< quadratic local interpolation (previous & current position)
                        cg_descent,                     ///< CG_DESCENT
                };

                ///
                /// \brief line-search strategy
                ///
                enum class strategy
                {
                        backtrack_armijo,               ///< backtracking with sufficient decrease (Armijo)
                        backtrack_wolfe,                ///< + backtracking with suficient curvature (Wolfe)
                        backtrack_swolfe,               ///< + backtracking with sufficient curvature (strong Wolfe)
                        cg_descent,                     ///< CG_DESCENT (regular and approximation Wolfe)
                        more_thuente,                   ///< More & Thunte (strong Wolfe)
                };

                ///
                /// \brief constructor
                ///
                lsearch_t(const initializer, const strategy, const scalar_t c1, const scalar_t c2);

                ///
                /// \brief update the current state
                ///
                bool operator()(solver_state_t& state);

        private:

                // attributes
                std::unique_ptr<lsearch_init_t>         m_initializer;  ///<
                std::unique_ptr<lsearch_strategy_t>     m_strategy;     ///<
        };

        template <>
        inline enum_map_t<lsearch_t::initializer> enum_string<lsearch_t::initializer>()
        {
                return
                {
                        { lsearch_t::initializer::unit,                 "unit" },
                        { lsearch_t::initializer::linear,               "linear" },
                        { lsearch_t::initializer::quadratic,            "quadratic" },
                        { lsearch_t::initializer::cg_descent,           "cg-descent" }
                };
        }

        template <>
        inline enum_map_t<lsearch_t::strategy> enum_string<lsearch_t::strategy>()
        {
                return
                {
                        { lsearch_t::strategy::backtrack_armijo,        "back-armijo" },
                        { lsearch_t::strategy::backtrack_wolfe,         "back-wolfe" },
                        { lsearch_t::strategy::backtrack_swolfe,        "back-swolfe" },
                        { lsearch_t::strategy::cg_descent,              "cg-descent" },
                        { lsearch_t::strategy::more_thuente,            "more-thuente" }
                };
        }
}
