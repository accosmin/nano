#pragma once

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief secant interpolation in the [step0, step1] line-search interval
                ///     (Nocedal & Wright (numerical optimization 2nd) @ p.59)
                ///
                template
                <
                        typename tstep,

                        typename tscalar = typename tstep::tscalar
                >
                tscalar ls_secant(const tstep& step0, const tstep& step1)
                {
                        return  (step0.alpha() * step1.gphi() - step1.alpha() * step0.gphi()) /
                                (step1.gphi() - step0.gphi());
                }
        }
}

