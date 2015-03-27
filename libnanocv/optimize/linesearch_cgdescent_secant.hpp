#pragma once

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief [a, b] line-search interval secant interpolation (see CG_DESCENT)
                ///
                template
                <
                        typename tstep
                >
                tstep cgdescent_secant(const tstep& a, const tstep& b)
                {
                        tstep c = a;
                        c.reset_with_grad(
                                (a.alpha() * b.gphi() - b.alpha() * a.gphi()) /
                                (b.gphi() - a.gphi()));

                        return c;
                }
        }
}

