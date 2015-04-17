#pragma once

#include "types.h"
#include "linesearch_step.hpp"
#include "linesearch_cubic.hpp"
#include "linesearch_bisection.hpp"
#include "linesearch_quadratic.hpp"
#include <vector>

namespace ncv
{
        namespace optim
        {
                ///
                /// \brief interpolation-based line-search (More & Thuente -like?!),
                ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.60
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar,
                        typename tsize = typename tstep::tsize
                >
                class linesearch_interpolation_t
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        linesearch_interpolation_t()
                        {
                        }

                        ///
                        /// \brief compute the current step size
                        ///
                        tstep operator()(
                                const ls_strategy strategy, const tscalar c1, const tscalar c2,
                                const tstep& step0, const tscalar t0,
                                const tsize max_iters = 64) const
                        {
                                // previous step
                                tstep stepp = step0;

                                // current step
                                tstep stept = step0;
                                tscalar t = t0;

                                for (tsize i = 1; i <= max_iters; i ++)
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

                                        if (stept.gphi() >= tscalar(0))
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
                        tstep zoom(
                                const ls_strategy strategy, const tscalar c1, const tscalar c2,
                                const tstep& step0, tstep steplo, tstep stephi,
                                const tsize max_iters = 64) const
                        {
                                tstep stept(step0);
                                tscalar t;

                                for (   size_t i = 1; i <= max_iters &&
                                        std::fabs(steplo.alpha() - stephi.alpha()) > stept.minimum(); i ++)
                                {
                                        // try various interpolation methods
                                        const auto tb = ls_bisection(steplo, stephi);
                                        const auto tq = ls_quadratic(steplo, stephi);
                                        const auto tc = ls_cubic(steplo, stephi);

                                        t = tb;

                                        std::vector<tscalar> trials;
                                        trials.push_back(tb);
                                        trials.push_back(tq);
                                        trials.push_back(tc.first);
                                        trials.push_back(tc.second);

                                        // choose the valid interpolation step closest to the minimum value step
                                        const tscalar tmin = std::min(steplo.alpha(), stephi.alpha());
                                        const tscalar tmax = std::max(steplo.alpha(), stephi.alpha());
                                        const tscalar teps = (tmax - tmin) / 20;

                                        tscalar best_dist = std::numeric_limits<tscalar>::max();
                                        for (const auto tt : trials)
                                        {
                                                if (std::isfinite(tt) && tmin + teps < tt && tt < tmax - teps)
                                                {
                                                        const tscalar dist = std::fabs(tt - steplo.alpha());
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
}

