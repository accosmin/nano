#pragma once

#include <cassert>

namespace tensor
{
        ///
        /// \brief running exponential average (aka momentum) for Eigen vectors with zero-bias correction
        ///     see "Adam: A method for stochastic optimization", by Diederik P. Kingma & Jimmy Lei Ba
        ///
        template <typename tvector, typename tscalar = typename tvector::Scalar>
        struct momentum_t
        {
                template <typename tsize>
                momentum_t(const tscalar momentum, const tsize dimensions) :
                        m_momentum(momentum),
                        m_correction(1),
                        m_value(std::move(tvector::Zero(dimensions)))
                {
                        assert(momentum > 0);
                        assert(momentum < 1);
                }

                template <typename texpression>
                void update(texpression&& value)
                {
                        m_value.array() = m_momentum * m_value.array() + (1 - m_momentum) * value.array();
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
                tvector         m_value;        ///< running exponential average
        };
}

