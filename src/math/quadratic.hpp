#pragma once

#include <cmath>

namespace math
{
        ///
        /// \brief quadratic: a x^2 + b x + c
        ///
        template
        <
                typename tscalar
        >
        class quadratic_t
        {
        public:
                ///
                /// \brief constructor
                ///
                quadratic_t(const tscalar a, const tscalar b, const tscalar c)
                        :       m_a(a),
                          m_b(b),
                          m_c(c)
                {
                }

                ///
                /// \brief quadratic interpolation of the [x0, x1] interval, using the function values & gradients:
                ///     [x0, f0 = f(x0), g0 = f'(x0)]
                ///     [x1, f1 = f(x1)]
                ///
                quadratic_t(const tscalar x0, const tscalar f0, const tscalar g0,
                            const tscalar x1, const tscalar f1)
                {
                        m_a = (g0 - (f0 - f1) / (x0 - x1)) / (x0 - x1);
                        m_b = g0 - 2 * m_a * x0;
                        m_c = f0 - m_b * x0 - m_a * x0 * x0;
                }

                ///
                /// \brief compute the extremum point
                ///
                void extremum(tscalar& min) const
                {
                        min = -m_b / (2 * m_a);
                }

                ///
                /// \brief compute the value for the input [x]
                ///
                tscalar value(const tscalar x) const
                {
                        return m_a * x * x + m_b * x + m_c;
                }

                ///
                /// \brief compute the gradient for the input [x]
                ///
                tscalar gradient(const tscalar x) const
                {
                        return 2 * m_a * x + m_b;
                }

                ///
                /// \brief check validity
                ///
                operator bool() const
                {
                        return  std::isfinite(m_a) &&
                                std::isfinite(m_b) &&
                                std::isfinite(m_c);
                }

                ///
                /// \brief access functions
                ///
                tscalar a() const { return m_a; }
                tscalar b() const { return m_b; }
                tscalar c() const { return m_c; }

        private:

                // attributes
                tscalar         m_a;
                tscalar         m_b;
                tscalar         m_c;
        };
}

