#pragma once

namespace math
{
        ///
        /// \brief running average for scalars
        ///
        template
        <
                typename tscalar
        >
        class average_scalar_t
        {
        public:

                ///
                /// \brief constructor
                ///
                average_scalar_t()
                        :       m_count(0),
                                m_value(0)
                {
                }

                ///
                /// \brief update the running average with a new value
                ///
                void update(const tscalar value)
                {
			constexpr tscalar one = 1;
                        m_value = (m_value * m_count + value) / (m_count + one);
                        m_count = m_count + one;
                }

                ///
                /// \brief retrieve the current average
                ///
                tscalar value() const
                {
                        return m_value;
                }

        private:

                tscalar         m_count;
                tscalar         m_value;
        };

	///
        /// \brief running average for Eigen vectors
        ///
        template
        <
                typename tvector,
		typename tscalar = typename tvector::Scalar
        >
        class average_vector_t
        {
        public:

                ///
                /// \brief constructor
                ///
                template
                <
                        typename tsize
                >
                explicit average_vector_t(const tsize dimensions)
                        :       m_count(0),
                                m_value(tvector::Zero(dimensions))
                {
                }

                ///
                /// \brief update the running average with a new value
                ///
                void update(const tvector& value)
                {
			constexpr tscalar one = 1;
                        m_value.noalias() = (m_value * m_count + value) / (m_count + one);
			m_count = m_count + one;
                }

                ///
                /// \brief retrieve the current average
                ///
                const tvector& value() const
                {
                        return m_value;
                }

        private:

                tscalar         m_count;
                tvector         m_value;
        };
}

