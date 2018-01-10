#pragma once

#include <cassert>
#include <type_traits>

namespace nano
{
        ///
        /// \brief running exponential average (aka momentum) for scalars with zero-bias correction
        ///     see "Adam: A method for stochastic optimization", by Diederik P. Kingma & Jimmy Lei Ba
        ///
        template <typename tscalar, typename = typename std::is_floating_point<tscalar>::type>
        struct momentum1_t
        {
                momentum1_t(const tscalar momentum) :
                        m_momentum(momentum),
                        m_correction(1),
                        m_value(0)
                {
                        assert(momentum > 0);
                        assert(momentum < 1);
                }

                void update(const tscalar value)
                {
                        m_value = m_momentum * m_value + (1 - m_momentum) * value;
                        m_correction *= m_momentum;
                }

                auto value() const
                {
                        const auto correction = (m_correction == 1) ? tscalar(1) : (tscalar(1) - m_correction);
                        assert(correction > 0);
                        return m_value * (1 / correction);
                }

        private:

                tscalar         m_momentum;     ///<
                tscalar         m_correction;   ///< zero-bias correction
                tscalar         m_value;        ///< running exponential average
        };
}

