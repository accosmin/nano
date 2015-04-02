#pragma once

#include "linesearch_cgdescent_updateU.hpp"

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief [a, b] line-search interval update (see CG_DESCENT)
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar,
                        typename tsize = typename tstep::tsize
                >
                std::pair<tstep, tstep> cgdescent_update(const tstep& step0,
                        const tstep& a, const tstep& b, tstep c,
                        const tscalar epsilon,
                        const tscalar theta)
                {
                        if (!c || c.alpha() <= a.alpha() || c.alpha() >= b.alpha())
                        {
                                return std::make_pair(a, b);
                        }

                        else if (c.gphi() >= 0)
                        {
                                return std::make_pair(a, c);
                        }

                        else if (c.phi() <= c.approx_phi(epsilon))
                        {
                                return std::make_pair(c, b);
                        }

                        else
                        {
                                return cgdescent_updateU(step0, a, c, epsilon, theta);
                        }
                }
        }
}

