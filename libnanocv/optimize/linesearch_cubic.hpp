#pragma once

#include <cmath>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief cubic interpolation in the [step0, step1] line-search interval
                ///     Nocedal & Wright (numerical optimization 2nd), p.59
                ///
                template
                <
                        typename tstep,

                        typename tscalar = typename tstep::tscalar
                >
                tscalar ls_cubic(const tstep& step0, const tstep& step1)
                {
                        const tscalar sign = (step1.alpha() > step0.alpha()) ? 1 : -1;

                        const tscalar d0 = (step0.phi() - step1.phi()) / (step0.alpha() - step1.alpha());
                        const tscalar d1 = step0.gphi() + step1.gphi() - 3 * d0;
                        const tscalar d2 = sign * std::sqrt(d1 * d1 - step0.gphi() * step1.gphi());

                        return  step1.alpha() -
                                (step1.alpha() - step0.alpha()) * (step1.gphi() + d2 - d1) /
                                (step1.gphi() - step0.gphi() + 2 * d2);
                }
        }
}

