#pragma once

#include <cmath>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief cubic interpolation in the [alpha0, alpha1] interval
                ///     (Nocedal & Wright (numerical optimization 2nd) @ p.59)
                ///
                template
                <
                        typename tscalar
                >
                tscalar ls_cubic(
                        const tscalar alpha0, const tscalar phi_alpha0, const tscalar gphi_alpha0,
                        const tscalar alpha1, const tscalar phi_alpha1, const tscalar gphi_alpha1)
                {
                        const tscalar sign = (alpha1 > alpha0) ? 1 : -1;

                        const tscalar d1 = gphi_alpha0 + gphi_alpha1 - 3 * (phi_alpha0 - phi_alpha1) / (alpha0 - alpha1);
                        const tscalar d2 = sign * std::sqrt(d1 * d1 - gphi_alpha0 * gphi_alpha1);

                        return alpha1 - (alpha1 - alpha0) * (gphi_alpha1 + d2 - d1) /
                                        (gphi_alpha1 - gphi_alpha0 + 2 * d2);
                }
        }
}

