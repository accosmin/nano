#pragma once

#include "linesearch_cgdescent_secant.hpp"
#include "linesearch_cgdescent_update.hpp"

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief [a, b] line-search interval double secant update (see CG_DESCENT)
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar,
                        typename tsize = typename tstep::tsize
                >
                std::pair<tstep, tstep> cgdescent_secant2(const tstep& a, const tstep& b,
                        tscalar epsilon, tscalar theta,
                        const tsize max_iters = 64)
                {
                        // (Hager & Zhang 2005, p. 15)
                        const tstep c = cgdescent_secant(a, b);

                        tstep A(a), B(b);
                        std::tie(A, B) = cgdescent_update(a, b, c, epsilon, theta, max_iters);

                        if (std::fabs(c.alpha() - A.alpha()) < std::numeric_limits<tscalar>::epsilon())
                        {
                                return cgdescent_update(A, B, cgdescent_secant(a, A), epsilon, theta, max_iters);
                        }

                        else if (std::fabs(c.alpha() - B.alpha()) < std::numeric_limits<tscalar>::epsilon())
                        {
                                return cgdescent_update(A, B, cgdescent_secant(b, B), epsilon, theta, max_iters);
                        }

                        else
                        {
                                return std::make_pair(A, B);
                        }
                }
        }
}

