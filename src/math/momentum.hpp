#pragma once

#include <cassert>

namespace math
{
        template
        <
                typename tscalar
        >
        struct momentum_t
        {
                momentum_t(const tscalar momentum) : m_momentum(momentum)
                {
                        assert(momentum > 0);
                        assert(momentum < 1);
                }

                template
                <
                        typename tvalue
                >
                auto update(const tvalue& avg_value, const tvalue& value) const
                {
                        constexpr tscalar one = 1;
                        return avg_value * m_momentum + value * (one - m_momentum);
                }

                tscalar         m_momentum;
        };

        ///
        /// \brief running exponential average for scalars using a fixed momentum
        ///
        template
        <
                typename tscalar,
                typename tbase = momentum_t<tscalar>
        >
        class momentum_scalar_t : private tbase
        {
        public:

                ///
                /// \brief constructor
                ///
		momentum_scalar_t(const tscalar momentum, const tscalar initial)
                        :       tbase(momentum),
                                m_value(initial)
                {
                }

                ///
                /// \brief update the running geometric average with a new value
                ///
                void update(const tscalar value)
                {
                        m_value = tbase::update(m_value, value);
                }

                ///
                /// \brief retrieve the current average
                ///
                tscalar value() const
                {
                        return m_value;
                }

        private:

                tscalar         m_value;
        };

	///
        /// \brief running exponential average for Eigen vectors using a fixed momentum
        ///
        template
        <
                typename tvector,
                typename tscalar = typename tvector::Scalar,
                typename tbase = momentum_t<tscalar>
        >
        class momentum_vector_t : private tbase
        {
        public:

                ///
                /// \brief constructor
                ///
                momentum_vector_t(const tscalar momentum, const tvector& initial)
                        :       tbase(momentum),
                                m_value(initial)
                {
                }

                ///
                /// \brief update the running average with a new value
                ///
                void update(const tvector& value)
                {
                        m_value.noalias() = tbase::update(m_value, value);
                }

                ///
                /// \brief retrieve the current average
                ///
                const tvector& value() const
                {
                        return m_value;
                }

        private:

                tvector         m_value;
        };
}

