#pragma once

namespace ncv
{
        namespace optim
        {
                ///
                /// \brief cubic interpolation of the [x0, x1] interval, using the function values & gradients:
                ///     [x0, f0 = f(x0), g0 = f'(x0)]
                ///     [x1, f1 = f(x1), g1 = f'(x1)]
                ///
                template
                <
                        typename tscalar
                >
                void cubic(
                        const tscalar x0, const tscalar f0, const tscalar g0,
                        const tscalar x1, const tscalar f1, const tscalar g1,
                        tscalar* poly_a = nullptr,
                        tscalar* poly_b = nullptr,
                        tscalar* poly_c = nullptr,
                        tscalar* poly_d = nullptr)
                {
                        // a x^3 + b x^2 + c x + d
                        const tscalar a = ((g0 + g1) / 2 - (f0 - f1) / (x0 - x1)) / (x0 * x0 + x0 * x1 + x1 * x1);
                        const tscalar b = ((g0 - g1) / (x0 - x1) - 3 * a * (x0 + x1)) / 2;
                        const tscalar c = g0 - 2 * b * x0 - 3 * a * x0 * x0;
                        const tscalar d = f0 - c * x0 - b * x0 * x0 - a * x0 * x0 * x0;

                        // retrieve polynomial coefficients
                        if (poly_a) { *poly_a = a; }
                        if (poly_b) { *poly_b = b; }
                        if (poly_c) { *poly_c = c; }
                        if (poly_d) { *poly_d = d; }
                }
        }
}

