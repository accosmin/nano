#pragma once

#include <limits>
#include <cassert>
#include "linesearch.h"

namespace ncv
{
        namespace optimize
        {
                template
                <
                        typename tproblem,

                        // dependent types
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tvector = typename tproblem::tvector,
                        typename tstate = typename tproblem::tstate
                >
                class linesearch_strategy_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        linesearch_strategy_t(
                                        ls_criterion criterion, ls_strategy strategy,
                                        tscalar c1 = 1e-4, tscalar c2 = 0.1)
                                :       m_criterion(criterion),
                                        m_strategy(strategy),
                                        m_c1(c1),
                                        m_c2(c2)
                        {
                        }

                        ///
                        /// \brief update the current state
                        ///
                        bool update(const tproblem& problem, tscalar t0, tstate& state) const
                        {
                                assert(m_c1 < m_c2);
                                assert(m_c1 > tscalar(0) && m_c1 < tscalar(1));
                                assert(m_c2 > tscalar(0) && m_c2 < tscalar(1));

                                const tscalar eps = std::numeric_limits<tscalar>::epsilon();
                                const tscalar tmin = eps;
                                const tscalar tmax = tscalar(1) / eps;

                                // check descent direction
                                const tscalar dg0 = state.d.dot(state.g);
                                if (dg0 > eps)
                                {
                                        return 0;
                                }

                                // check valid initial step
                                if (t0 < eps)
                                {
                                        return 0;
                                }

                                tscalar ft;
                                tvector gt;

                                const tscalar t = step(problem, t0, tmin, tmax, state, dg0, ft, gt);
                                if (t < std::numeric_limits<tscalar>::epsilon())
                                {
                                        // failed to find a suitable line-search step
                                        return false;
                                }
                                else
                                {
                                        // OK, update the current state
                                        state.update(problem, t, ft, gt);
                                        return true;
                                }
                        }

                private:

                        tscalar step(const tproblem& problem, tscalar t, const tscalar tmin, const tscalar tmax,
                                const tstate& state, const tscalar dg0, tscalar& ft, tvector& gt) const
                        {
                                switch (m_strategy)
                                {
                                case ls_strategy::backtracking:
                                        return step_backtracking(problem, t, tmin, tmax, state, dg0, ft, gt);

                                case ls_strategy::nocedal:
                                default:
                                        return step_nocedal(problem, t, tmin, tmax, state, dg0, ft, gt);
                                }
                        }

                        tscalar step_backtracking(const tproblem& problem,
                                tscalar t, const tscalar tmin, const tscalar tmax,
                                const tstate& state, const tscalar dg0,
                                tscalar& ft, tvector& gt, tsize max_iters = 64) const
                        {
                                const tscalar dec = 0.5;
                                const tscalar inc = 2.1;

                                // implementation inspired by libLBFGS
                                for (tsize i = 0; i < max_iters && t > tmin && t < tmax; i ++)
                                {
                                        ft = problem(state.x + t * state.d, gt);

                                        // check Armijo condition
                                        if (ft > state.f + t * m_c1 * dg0)
                                        {
                                                t *= dec;
                                        }
                                        else
                                        {
                                                if (m_criterion == ls_criterion::armijo)
                                                {
                                                        return t;
                                                }

                                                // check Wolfe condition
                                                const tscalar dgt = state.d.dot(gt);
                                                if (dgt < m_c2 * dg0)
                                                {
                                                        t *= inc;
                                                }
                                                else
                                                {
                                                        if (m_criterion == ls_criterion::wolfe)
                                                        {
                                                                return t;
                                                        }

                                                        // check strong Wolfe condition
                                                        if (dgt > - m_c2 * dg0)
                                                        {
                                                                t *= dec;
                                                        }
                                                        else
                                                        {
                                                                return t;
                                                        }
                                                }
                                        }
                                }

                                // OK, give up
                                return 0;
                        }

                        tscalar step_nocedal(const tproblem& problem,
                                tscalar t, const tscalar tmin, const tscalar tmax,
                                const tstate& state, const tscalar dg0,
                                tscalar& ft, tvector& gt, tsize max_iters = 64) const
                        {
                                tscalar t0 = 0, ft0 = std::numeric_limits<tscalar>::max();

                                /// \todo Armijo, Wolfe & strong-Wolfe conditions

                                // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
                                for (tsize i = 0; i < max_iters; i ++)
                                {
                                        // check sufficient decrease
                                        ft = problem(state.x + t * state.d, gt);
                                        if (ft > state.f + m_c1 * t * dg0 || (ft >= ft0 && i > 0))
                                        {
                                                return ls_zoom(problem, state, dg0, ft, gt, t0, t, ft0, ft);
                                        }

                                        // check curvature
                                        const tscalar dg1 = gt.dot(state.d);
                                        if (std::fabs(dg1) <= -m_c2 * dg0)
                                        {
                                                return t;
                                        }

                                        if (dg1 >= 0)
                                        {
                                                return ls_zoom(problem, state, dg0, ft, gt, t, t0, ft, ft0);
                                        }

                                        t0 = t;
                                        ft0 = ft;
                                        t = std::min(tmax, t * 3);
                                }

                                // OK, give up
                                return 0;
                        }

                        tscalar ls_zoom(const tproblem& problem, const tstate& state, const tscalar dg0,
                                tscalar& ft, tvector& gt,
                                tscalar tlo, tscalar thi, tscalar ftlo, tscalar fthi, size_t max_iters = 10) const
                        {
                                // (Nocedal & Wright (numerical optimization 2nd) @ p.60)
        //                        while (std::fabs(thi - tlo) > std::numeric_limits<tscalar>::epsilon())
                                for (size_t i = 0; i < max_iters; i ++)
                                {
                                        /// \todo cubic interpolation
                                        const tscalar t = (tlo + thi) / 2;

                                        // check sufficient decrease
                                        ft = problem(state.x + t * state.d, gt);
                                        if (ft > state.f + m_c1 * t * dg0 || ft >= ftlo)
                                        {
                                                thi = t;
                                                fthi = ft;
                                        }

                                        // check curvature
                                        else
                                        {
                                                const tscalar dg1 = gt.dot(state.d);
                                                if (std::fabs(dg1) <= -m_c2 * dg0)
                                                {
                                                        return t;
                                                }

                                                if (dg1 * (thi - tlo) >= 0)
                                                {
                                                        thi = tlo;
                                                        fthi = ftlo;
                                                }

                                                tlo = t;
                                                ftlo = ft;
                                        }
                                }

                                // OK, give up
                                return 0;
                        }

                private:

                        // attributes
                        ls_criterion    m_criterion;    ///<
                        ls_strategy     m_strategy;     ///<
                        tscalar         m_c1;           ///< sufficient decrease rate
                        tscalar         m_c2;           ///< sufficient curvature
                };
        }
}

