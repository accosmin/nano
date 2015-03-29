#pragma once

#include "linesearch_zoom.hpp"

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
                tstep ls_interpolation(
                        const ls_strategy strategy, const tscalar c1, const tscalar c2,
                        const tstep& step0, const tscalar t0, const tsize max_iters = 64)
                {
                        // previous step
                        tstep stepp = step0;

                        // current step
                        tstep stept = step0;
                        tscalar t = t0;

                        // Nocedal & Wright (numerical optimization 2nd, p.60
                        for (tsize i = 1; i <= max_iters; i ++)
                        {
                                // check sufficient decrease
                                if (!stept.reset_with_grad(t))
                                {
                                        break;
                                }

                                if (!stept.has_armijo(c1) || (stept.func() >= stepp.func() && i > 1))
                                {
                                        return ls_zoom(strategy, c1, c2, step0, stepp, stept);
                                }

                                // check curvature
                                if (stept.has_strong_wolfe(c2))
                                {
                                        return stept.setup();
                                }

                                if (stept.gphi() >= tscalar(0))
                                {
                                        return ls_zoom(strategy, c1, c2, step0, stept, stepp);
                                }

                                stepp = stept;
                                t = std::min(stept.maximum(), t * 3);
                        }

                        // NOK, give up
                        return stept.phi() < step0.phi() ? stept : step0;
                }
        }
}

