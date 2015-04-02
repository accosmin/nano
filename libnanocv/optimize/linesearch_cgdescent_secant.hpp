#pragma once

#include <cmath>
#include <limits>

namespace ncv
{
        namespace optimize
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
                        const auto div = b.gphi() - a.gphi();

                        if (std::fabs(div) > std::numeric_limits<tscalar>::epsilon())
                        {
                                const auto t = (a.alpha() * b.gphi() - b.alpha() * a.gphi()) / div;

                                if (std::isfinite(t))
                                {
                                        tstep c = a;
                                        c.reset(t);
                                        return c;
                                }
                        }

                        return a;
                }
        }
}

