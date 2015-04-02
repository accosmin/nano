#pragma once

#include <cmath>
#include <utility>

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
                std::pair<tstep, tstep> cgdescent_updateU(tstep a, tstep b,
                        const tscalar epsilon,
                        const tscalar theta,
                        const tsize max_iters = 128)
                {
                        tstep c(a);
                        for (tsize i = 0; i < max_iters && (b.alpha() - a.alpha()) > a.minimum(); i ++)
                        {
                                c.reset((1 - theta) * a.alpha() + theta * b.alpha());

                                if (c.gphi() >= 0)
                                {
                                        return std::make_pair(a, c);
                                }

                                else if (c.phi() <= c.approx_phi(epsilon))
                                {
                                        a = c;
                                }

                                else
                                {
                                        b = c;
                                }
                        }

                        // NOK, give up
                        return std::make_pair(c, c);
                }
        }
}

