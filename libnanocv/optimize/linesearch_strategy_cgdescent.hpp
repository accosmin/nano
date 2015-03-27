#pragma once

#include "linesearch.h"
#include "linesearch_step.hpp"
#include "linesearch_cgdescent_secant2.hpp"

namespace ncv
{
        namespace optimize
        {
                template
                <
                        typename tproblem,
                        typename tscalar = typename tproblem::tscalar,
                        typename tsize = typename tproblem::tsize,
                        typename tstep = ls_step_t<tproblem>
                >
                tscalar ls_cgdescent(
                        const ls_strategy strategy, const tscalar c1, const tscalar c2,
                        tscalar t, tstep& stept, tsize max_iters = 64)
                {
                        const tscalar epsilon = tscalar(1e-6);
                        const tscalar theta = tscalar(0.5);
                        const tscalar gamma = tscalar(0.66);

                        tstep a(stept);
                        tstep b(stept);
                        b.reset_with_grad(t);

                        // CG_DESCENT (Hager & Zhang 2005, p. 15)
                        for (tsize i = 0; i < max_iters && std::fabs(a.alpha() - b.alpha()) > a.minimum(); i ++)
                        {
                                // check Wolfe & approximate Wolfe condition
                                if (b.has_approx_wolfe(c2, epsilon))
                                {
                                        return b.setup();
                                }

                                // secant interpolation
                                tstep A(a), B(a);
                                std::tie(A, B) = cgdescent_secant2(a, b, epsilon, theta);

                                // update search interval
                                if ((B.alpha() - A.alpha()) > gamma * (a.alpha() - b.alpha()))
                                {
                                        tstep c(a);
                                        c.reset_with_grad((A.alpha() + B.alpha()) / 2);
                                        std::tie(a, b) = cgdescent_update(A, B, c, epsilon, theta);
                                }
                                else
                                {
                                        a = A;
                                        b = B;
                                }
                        }

                        // OK, give up
                        return 0;
                }
        }
}

