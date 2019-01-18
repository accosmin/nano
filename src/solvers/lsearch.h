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
                /// \brief cubic interpolation following the notation from:
                ///     see "Line Search Algorithms with Guaranteed Sufficient Decrease",
                ///     by Jorge J. More and David J. Thuente
                ///
                static auto cubic(
                        const scalar_t u, const scalar_t fu, const scalar_t gu,
                        const scalar_t v, const scalar_t fv, const scalar_t gv)
                {
                        // fit cubic: q(x) = a*x^3 + b*x^2 + c*x + d
                        //      given: q(u) = fu, q'(u) = gu
                        //      given: q(v) = fv, q'(v) = gv
                        // minimizer: solution of 3*a*x^2 + 2*b*x + c = 0
                        // see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
                        const auto d1 = gu + gv - 3 * (fu - fv) / (u - v);
                        const auto d2 = (v > u ? +1 : -1) * std::sqrt(d1 * d1 - gu * gv);
                        return v - (v - u) * (gv + d2 - d1) / (gv - gu + 2 * d2);
                }

                ///
                /// \brief qudratic interpolation following the notation from:
                ///     see "Line Search Algorithms with Guaranteed Sufficient Decrease",
                ///     by Jorge J. More and David J. Thuente
                ///
                static auto quadratic(
                        const scalar_t u, const scalar_t fu, const scalar_t gu,
                        const scalar_t v, const scalar_t fv)
                {
                        // fit quadratic: q(x) = a*x^2 + b*x + c
                        //      given: q(u) = fu, q'(u) = gu
                        //      given: q(v) = fv
                        // minimizer: -b/2a
                        return u - gu * (u - v) * (u - v) / (2 * (gu * (u - v) - (fu - fv)));
                }

                ///
                /// \brief qudratic interpolation following the notation from:
                ///     see "Line Search Algorithms with Guaranteed Sufficient Decrease",
                ///     by Jorge J. More and David J. Thuente
                ///
                static auto quadratic(
                        const scalar_t u, const scalar_t gu,
                        const scalar_t v, const scalar_t gv)
                {
                        // fit quadratic: q(x) = a*x^2 + b*x + c
                        //      given: q'(u) = gu
                        //      given: q'(v) = gv
                        // minimizer: -b/2a
                        return (v * gu - u * gv) / (gu - gv);
                }

                ///
                /// \brief qudratic interpolation following the notation from:
                ///     see "Line Search Algorithms with Guaranteed Sufficient Decrease",
                ///     by Jorge J. More and David J. Thuente
                ///
                static auto bisection(
                        const scalar_t u, const scalar_t v)
                {
                        // minimizer: (u+v)/2
                        return (u + v) / 2;
                }

                ///
                /// \brief interpolate closest to the minimum value point following the notation from:
                ///     see "Line Search Algorithms with Guaranteed Sufficient Decrease",
                ///     by Jorge J. More and David J. Thuente
                ///
                static auto interpolate(
                        const scalar_t u, const scalar_t fu, const scalar_t gu,
                        const scalar_t v, const scalar_t fv, const scalar_t gv)
                {
                        // NB: this doesn't assume the points u and v are sorted!
                        // NB: this assumes that u is the point with the minimum value!
                        const auto tc = cubic(u, fu, gu, v, fv, gv);
                        const auto tq = quadratic(u, gu, v, gv);
                        const auto ts = quadratic(u, fu, gu, v, fv);
                        const auto tb = bisection(u, v);

                        scalar_t tt[4];
                        int tcount = 0;

                        const auto op_add = [&] (const auto t)
                        {
                                if (std::isfinite(t) && std::min(u, v) < t && t < std::max(u, v))
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
                                return std::fabs(t1 - u) < std::fabs(t2 - u);
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
