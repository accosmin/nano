#pragma once

#include "types.h"
#include "ls_cubic.hpp"
#include "ls_bisection.hpp"
#include "ls_quadratic.hpp"

namespace nano
{
        ///
        /// \brief interpolation-based line-search (More & Thuente -like?!),
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.60
        ///
        class ls_interpolate_t
        {
        public:

                ///
                /// \brief constructor
                ///
                ls_interpolate_t()
                {
                }

                ///
                /// \brief compute the current step size
                ///
                ls_step_t operator()(
                        const ls_strategy strategy, const scalar_t c1, const scalar_t c2,
                        const ls_step_t& step0, const scalar_t t0,
                        const int max_iters = 64) const
                {
                        // previous step
                        ls_step_t stepp = step0;

                        // current step
                        ls_step_t stept = step0;
                        scalar_t t = t0;

                        for (int i = 1; i <= max_iters; i ++)
                        {
                                // check sufficient decrease
                                if (!stept.reset(t))
                                {
                                        return step0;
                                }

                                if (!stept.has_armijo(c1) || (stept.func() >= stepp.func() && i > 1))
                                {
                                        return zoom(strategy, c1, c2, step0, stepp, stept);
                                }

                                // check curvature
                                if (stept.has_strong_wolfe(c2))
                                {
                                        return stept;
                                }

                                if (stept.gphi() >= scalar_t(0))
                                {
                                        return zoom(strategy, c1, c2, step0, stept, stepp);
                                }

                                stepp = stept;
                                t = std::min(stept.maximum(), t * 3);
                        }

                        // NOK, give up
                        return step0;
                }

        private:

                ///
                /// \brief zoom-in in the bracketed interval,
                ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.60
                ///
                static ls_step_t zoom(
                        const ls_strategy, const scalar_t c1, const scalar_t c2,
                        const ls_step_t& step0, ls_step_t steplo, ls_step_t stephi,
                        const int max_iters = 64)
                {
                        ls_step_t stept(step0);
                        scalar_t t;

                        for (   int i = 1; i <= max_iters &&
                                std::fabs(steplo.alpha() - stephi.alpha()) > stept.minimum(); i ++)
                        {
                                // try various interpolation methods
                                const auto tb = ls_bisection(steplo, stephi);
                                const auto tq = ls_quadratic(steplo, stephi);
                                const auto tc = ls_cubic(steplo, stephi);

                                t = tb;

                                std::vector<scalar_t> trials;
                                trials.push_back(tb);
                                trials.push_back(tq);
                                trials.push_back(tc.first);
                                trials.push_back(tc.second);

                                // choose the valid interpolation step closest to the minimum value step
                                const scalar_t tmin = std::min(steplo.alpha(), stephi.alpha());
                                const scalar_t tmax = std::max(steplo.alpha(), stephi.alpha());
                                const scalar_t teps = (tmax - tmin) / 20;

                                scalar_t best_dist = std::numeric_limits<scalar_t>::max();
                                for (const auto tt : trials)
                                {
                                        if (std::isfinite(tt) && tmin + teps < tt && tt < tmax - teps)
                                        {
                                                const scalar_t dist = std::fabs(tt - steplo.alpha());
                                                if (dist < best_dist)
                                                {
                                                        best_dist = dist;
                                                        t = tt;
                                                }
                                        }
                                }

                                // check sufficient decrease
                                if (!stept.reset(t))
                                {
                                        return step0;
                                }

                                if (!stept.has_armijo(c1) || stept.phi() >= steplo.phi())
                                {
                                        stephi = stept;
                                }

                                // check curvature
                                else
                                {
                                        if (stept.has_strong_wolfe(c2))
                                        {
                                                return stept;
                                        }

                                        if (stept.gphi() * (stephi.alpha() - steplo.alpha()) >= 0)
                                        {
                                                stephi = steplo;
                                        }

                                        steplo = stept;
                                }
                        }

                        // NOK, give up
                        return step0;
                }
        };
}

