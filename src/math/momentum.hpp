#pragma once

#include <cassert>

namespace math
{
        ///
        /// \brief running geometric average for scalars using a fixed momentum
        ///
        template
        <
                typename tscalar
        >
        class momentum_scalar_t
        {
        public:

                ///
                /// \brief constructor
                ///
		momentum_scalar_t(const tscalar momentum, const tscalar initial)
                        :       m_momentum(momentum),
				m_value(initial)
                {
			assert(momentum > 0);
			assert(momentum < 1);
                }

                ///
                /// \brief update the running geometric average with a new value
                ///
                void update(const tscalar value)
                {
			constexpr tscalar one = 1;
                        m_value = m_value * m_momentum + value * (one - m_momentum);
                }

                ///
                /// \brief retrieve the current average
                ///
                tscalar value() const
                {
                        return m_value;
                }

        private:

                tscalar         m_momentum;
                tscalar         m_value;
        };

	///
        /// \brief running geometric average for Eigen vectors using a fixed momentum
        ///
        template
        <
                typename tvector,
		typename tscalar = typename tvector::Scalar
        >
        class momentum_vector_t
        {
        public:

                ///
                /// \brief constructor
                ///
                momentum_vector_t(const tscalar momentum, const tvector& initial)
                        :       m_momentum(momentum),
                                m_value(initial)
                {
			assert(momentum > 0);
			assert(momentum < 1);
                }

                ///
                /// \brief update the running average with a new value
                ///
                void update(const tvector& value)
                {
			constexpr tscalar one = 1;
                        m_value.noalias() = m_value * m_momentum + value * (one - m_momentum);
                }

                ///
                /// \brief retrieve the current average
                ///
                const tvector& value() const
                {
                        return m_value;
                }

        private:

                tscalar         m_momentum;
                tvector         m_value;
        };
}

