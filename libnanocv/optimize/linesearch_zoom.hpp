#pragma once

#include <algorithm>
#include "linesearch.h"
#include "linesearch_step.hpp"
#include "linesearch_cubic.hpp"

namespace ncv
{
        namespace optimize
        {
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

                        // Nocedal & Wright (numerical optimization 2nd), p.60
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
                                        t = 0.33 * tmin + 0.67 * tmax;
                                        break;
                                }

                                // check sufficient decrease
                                if (!stept.reset_with_grad(t))
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
                                                return stept.setup();
                                        }

                                        if (stept.gphi() * (stephi.alpha() - steplo.alpha()) >= 0)
                                        {
                                                stephi = steplo;
                                        }

                                        steplo = stept;
                                }
                        }

                        // NOK, give up
                        return stept.phi() < step0.phi() ? stept : step0;
                }
        }
}

