#pragma once

namespace ncv
{
        namespace optim
        {
                ///
                /// \brief quadratic interpolation of the [x0, x1] interval, using the function values & gradients:
                ///     [x0, f0 = f(x0), g0 = f'(x0)]
                ///     [x1, f1 = f(x1)]
                ///
                template
                <
                        typename tscalar
                >
                void quadratic(
                        const tscalar x0, const tscalar f0, const tscalar g0,
                        const tscalar x1, const tscalar f1,
                        tscalar* poly_a = nullptr,
                        tscalar* poly_b = nullptr,
                        tscalar* poly_c = nullptr)
                {
                        // a x^2 + b x + c
                        const tscalar a = (g0 - (f0 - f1) / (x0 - x1)) / (x0 - x1);
                        const tscalar b = g0 - 2 * a * x0;
                        const tscalar c = f0 - b * x0 - a * x0 * x0;

                        // retrieve polynomial coefficients
                        if (poly_a) { *poly_a = a; }
                        if (poly_b) { *poly_b = b; }
                        if (poly_c) { *poly_c = c; }
                }
        }
}

