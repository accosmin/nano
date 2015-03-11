#pragma once

#include <cmath>
#include <iostream>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief quadratic interpolation in the [0, alpha0] interval
                ///     (Nocedal & Wright (numerical optimization 2nd) @ p.58)
                ///
                template
                <
                        typename tscalar
                >
                tscalar ls_interp_quadratic(
                        const tscalar phi_zero, const tscalar gphi_zero,
                        const tscalar alpha0, const tscalar phi_alpha0)
                {
                        return -(gphi_zero * alpha0 * alpha0) /
                                (2 * (phi_alpha0 - phi_zero - gphi_zero * alpha0));
                }

                ///
                /// \brief cubic interpolation in the [0, alpha0] interval
                ///     (Nocedal & Wright (numerical optimization 2nd) @ p.58)
                ///
                template
                <
                        typename tscalar
                >
                tscalar ls_interp_cubic(
                        const tscalar phi_zero, const tscalar gphi_zero,
                        const tscalar alpha0, const tscalar phi_alpha0,
                        const tscalar alpha1, const tscalar phi_alpha1)
                {
                        const tscalar dalpha = 1 / (alpha1 - alpha0);

                        const tscalar u00 = dalpha / (alpha1 * alpha1), u01 = -dalpha / (alpha0 * alpha0);
                        const tscalar u10 = -u00 * alpha0, u11 = -u01 * alpha1;

                        const tscalar v0 = phi_alpha1 - phi_zero - gphi_zero * alpha1;
                        const tscalar v1 = phi_alpha0 - phi_zero - gphi_zero * alpha0;

                        const tscalar a = u00 * v0 + u01 * v1;
                        const tscalar b = u10 * v0 + u11 * v1;

                        if (b * b < 3 * a * gphi_zero)
                        {
                                std::cout << "NOK!" << std::endl;
                        }

                        return (-b + std::sqrt(b * b - 3 * a * gphi_zero)) / (3 * a);
                }
        }
}

