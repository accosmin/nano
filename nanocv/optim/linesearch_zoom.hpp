#pragma once

#include "types.h"
#include "linesearch_step.hpp"
#include "linesearch_cubic.hpp"
#include "linesearch_bisection.hpp"
#include "linesearch_quadratic.hpp"

namespace ncv
{
        namespace optim
        {
                ///
                /// \brief zoom-in in the bracketed interval,
                ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.60
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar,
                        typename tsize = typename tstep::tsize
                >
                tstep ls_zoom(
                        const ls_strategy strategy, const tscalar c1, const tscalar c2,
                        const tstep& step0, tstep steplo, tstep stephi,
                        const tsize max_iters = 64)
                {
                        tstep stept(step0);
                        tscalar t;

                        for (   size_t i = 1; i <= max_iters &&
                                std::fabs(steplo.alpha() - stephi.alpha()) > stept.minimum(); i ++)
                        {
                                const tscalar tmin = std::min(steplo.alpha(), stephi.alpha());
                                const tscalar tmax = std::max(steplo.alpha(), stephi.alpha());
                                const tscalar teps = stept.minimum();

                                switch (strategy)
                                {
                                case ls_strategy::interpolation_cubic:
                                        t = ls_cubic(steplo, stephi);
                                        if (std::isfinite(t) && tmin + teps < t && t < tmax - teps)
                                        {
                                                break;
                                        }
                                        // fallthrough!

                                case ls_strategy::interpolation_bisection:
                                default:
                                        t = (steplo.alpha() + stephi.alpha()) / 2;
                                        break;
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
        }
}

