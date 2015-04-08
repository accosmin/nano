#pragma once

#include <cmath>

namespace ncv
{
        namespace optim
        {
                ///
                /// \brief cubic interpolation of the [x0, x1] interval, using the function values & gradients:
                ///     [x0, f0 = f(x0), g0 = f'(x0)]
                ///     [x1, f1 = f(x1), g1 = f'(x1)]
                ///
                /// \note the resulting cubic is: a x^3 + b x^2 + c x + d
                ///
                template
                <
                        typename tscalar
                >
                void cubic(
                        const tscalar x0, const tscalar f0, const tscalar g0,
                        const tscalar x1, const tscalar f1, const tscalar g1,
                        tscalar& a, tscalar& b, tscalar& c, tscalar& d)
                {
                        a = ((g0 + g1) / 2 - (f0 - f1) / (x0 - x1)) / (x0 * x0 + x0 * x1 + x1 * x1);
                        b = ((g0 - g1) / (x0 - x1) - 3 * a * (x0 + x1)) / 2;
                        c = g0 - 2 * b * x0 - 3 * a * x0 * x0;
                        d = f0 - c * x0 - b * x0 * x0 - a * x0 * x0 * x0;
                }

                ///
                /// \brief compute the extremum points for a cubic
                ///
                template
                <
                        typename tscalar
                >
                void cubic_extremum(const tscalar a, const tscalar b, const tscalar c, const tscalar d,
                        tscalar& min1, tscalar& min2)
                {
                        const tscalar d1 = -2 * b;
                        const tscalar d2 = std::sqrt(4 * b * b - 12 * a * c);

                        min1 = (d1 - d2) / (6 * a);
                        min2 = (d1 + d2) / (6 * a);
                }
        }
}

