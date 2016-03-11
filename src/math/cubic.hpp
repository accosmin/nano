#pragma once

#include <cmath>
#include <limits>
#include <cassert>

namespace nano
{
        ///
        /// \brief cubic: a x^3 + b x^2 + c x + d
        ///
        template
        <
                typename tscalar
        >
        class cubic_t
        {
        public:
                ///
                /// \brief constructor
                ///
                cubic_t(const tscalar a, const tscalar b, const tscalar c, const tscalar d)
                        :       m_a(a),
                                m_b(b),
                                m_c(c),
                                m_d(d)
                {
                }

                ///
                /// \brief cubic interpolation of the [x0, x1] interval, using the function values & gradients:
                ///     [x0, f0 = f(x0), g0 = f'(x0)]
                ///     [x1, f1 = f(x1), g1 = f'(x1)]
                ///
                cubic_t(const tscalar x0, const tscalar f0, const tscalar g0,
                        const tscalar x1, const tscalar f1, const tscalar g1)
                        :       cubic_t(
                                std::numeric_limits<tscalar>::infinity(),
                                std::numeric_limits<tscalar>::infinity(),
                                std::numeric_limits<tscalar>::infinity(),
                                std::numeric_limits<tscalar>::infinity())

                {
                        if (    (x0 - x1) != tscalar(0) &&
                                (x0 * x0 - 2 * x0 * x1 + x1 * x1) != tscalar(0))
                        {
                                m_a = ((g0 + g1) - 2 * (f0 - f1) / (x0 - x1)) / (x0 * x0 - 2 * x0 * x1 + x1 * x1);
                                m_b = ((g0 - g1) / (x0 - x1) - 3 * m_a * (x0 + x1)) / 2;
                                m_c = g0 - 2 * m_b * x0 - 3 * m_a * x0 * x0;
                                m_d = f0 - m_c * x0 - m_b * x0 * x0 - m_a * x0 * x0 * x0;
                         }
                }

                ///
                /// \brief compute the extremum points
                ///
                void extremum(tscalar& min1, tscalar& min2) const
                {
                        assert(operator bool());

                        const tscalar d1 = -m_b;
                        const tscalar d2 = std::sqrt(m_b * m_b - 3 * m_a * m_c);

                        min1 = (d1 - d2) / (3 * m_a);
                        min2 = (d1 + d2) / (3 * m_a);
                }

                ///
                /// \brief compute the value for the input [x]
                ///
                tscalar value(const tscalar x) const
                {
                        return m_a * x * x * x + m_b * x * x + m_c * x + m_d;
                }

                ///
                /// \brief compute the gradient for the input [x]
                ///
                tscalar gradient(const tscalar x) const
                {
                        return 3 * m_a * x * x + 2 * m_b * x + m_c;
                }

                ///
                /// \brief check validity
                ///
                operator bool() const
                {
                        return  std::isfinite(m_a) &&
                                std::isfinite(m_b) &&
                                std::isfinite(m_c) &&
                                std::isfinite(m_d) &&
                                m_a != tscalar(0) &&
                                (m_b * m_b - 3 * m_a * m_c) >= tscalar(0);
                }

                ///
                /// \brief access functions
                ///
                tscalar a() const { return m_a; }
                tscalar b() const { return m_b; }
                tscalar c() const { return m_c; }
                tscalar d() const { return m_d; }

        private:

                // attributes
                tscalar         m_a;
                tscalar         m_b;
                tscalar         m_c;
                tscalar         m_d;
        };
}

