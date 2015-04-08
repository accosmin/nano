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
                /// \note the resulting quadratic is: a x^2 + b x + c
                ///
                template
                <
                        typename tscalar
                >
                void quadratic(
                        const tscalar x0, const tscalar f0, const tscalar g0,
                        const tscalar x1, const tscalar f1,
                        tscalar& a, tscalar& b, tscalar& c)
                {
                        a = (g0 - (f0 - f1) / (x0 - x1)) / (x0 - x1);
                        b = g0 - 2 * a * x0;
                        c = f0 - b * x0 - a * x0 * x0;
                }

                ///
                /// \brief compute the extremum point for a quadratic
                ///
                template
                <
                        typename tscalar
                >
                void quadratic_extremum(const tscalar a, const tscalar b, const tscalar c,
                        tscalar& min)
                {
                        min = -b / (2 * a);
                }

                ///
                /// \brief compute the quadratic value for the input [x]
                ///
                template
                <
                        typename tscalar
                >
                tscalar quadratic_value(const tscalar a, const tscalar b, const tscalar c,
                        const tscalar x)
                {
                        return a * x * x + b * x + c;
                }

                ///
                /// \brief compute the quadratic gradient for the input [x]
                ///
                template
                <
                        typename tscalar
                >
                tscalar quadratic_grad(const tscalar a, const tscalar b, const tscalar c,
                        const tscalar x)
                {
                        return 2 * a * x + b;
                }
        }
}

