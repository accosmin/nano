#pragma once

#include <cassert>

namespace nano
{
        ///
        /// \brief running exponential average (aka momentum) with zero-bias correction
        ///     see "Adam: A method for stochastic optimization", by Diederik P. Kingma & Jimmy Lei Ba
        ///
        template
        <
                typename tscalar,
                typename tvalue
        >
        class momentum_t
        {
        public:
                momentum_t(const tscalar momentum, tvalue&& initial) :
                        m_momentum(momentum),
                        m_correction(1),
                        m_value(std::move(initial))
                {
                        assert(momentum > 0);
                        assert(momentum < 1);
                }

                void update(const tvalue& value)
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
                tvalue          m_value;        ///< running average
        };


        ///
        /// \brief running exponential average for scalars using a fixed momentum
        ///
        template
        <
                typename tscalar,
                typename tbase = momentum_t<tscalar, tscalar>
        >
        struct momentum_scalar_t : public tbase
        {
                explicit momentum_scalar_t(const tscalar momentum) :
                        tbase(momentum, 0)
                {
                }
        };

        ///
        /// \brief running exponential average for Eigen vectors using a fixed momentum
        ///
        template
        <
                typename tvector,
                typename tscalar = typename tvector::Scalar,
                typename tbase = momentum_t<tscalar, tvector>
        >
        struct momentum_vector_t : public tbase
        {
                template
                <
                        typename tsize
                >
                momentum_vector_t(const tscalar momentum, const tsize dimensions) :
                        tbase(momentum, tvector::Zero(dimensions))
                {
                }
        };
}

