#pragma once

#include "cubic.hpp"
#include <cmath>

namespace ncv
{
        namespace optim
        {
                ///
                /// \brief cubic interpolation in the [step0, step1] line-search interval
                ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
                ///
                template
                <
                        typename tstep,
                        typename tscalar = typename tstep::tscalar
                >
                tscalar ls_cubic(const tstep& step0, const tstep& step1)
                {
                        const tscalar x0 = step0.alpha(), f0 = step0.phi(), g0 = step0.gphi();
                        const tscalar x1 = step1.alpha(), f1 = step1.phi(), g1 = step1.gphi();

                        tscalar a, b, c, d;
                        optim::cubic(x0, f0, g0, x1, f1, g1, a, b, c, d);

                        // OK, return minimum
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

