#pragma once

#include <cmath>

namespace ncv
{
        namespace optimize
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

                        // a x^3 + b x^2 + c x + d
                        const tscalar a = ((g0 + g1) / 2 - (f0 - f1) / (x0 - x1)) / (x0 * x0 + x0 * x1 + x1 * x1);
                        const tscalar b = ((g0 - g1) / (x0 - x1) - 3 * a * (x0 + x1)) / 2;
                        const tscalar c = g0 - 2 * b * x0 - 3 * a * x0 * x0;
                        const tscalar d = f0 - c * x0 - b * x0 * x0 - a * x0 * x0 * x0;

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

