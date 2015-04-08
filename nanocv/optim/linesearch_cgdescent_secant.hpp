#pragma once

#include <cmath>
#include <limits>

namespace ncv
{
        namespace optim
        {
                ///
                /// \brief [a, b] line-search interval secant interpolation (see CG_DESCENT)
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar
                >
                tstep cgdescent_secant(const tstep& a, const tstep& b)
                {
                        const auto t = (a.alpha() * b.gphi() - b.alpha() * a.gphi()) /
                                       (b.gphi() - a.gphi());

                        tstep c = a;
                        if (!c.reset(t))
                        {
                                return a;
                        }
                        else
                        {
                                return c;
                        }
                }
        }
}

