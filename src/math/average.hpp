#pragma once

namespace math
{
        template
        <
                typename tscalar
        >
        struct average_t
        {
                average_t() : m_count(0)
                {
                }

                template
                <
                        typename tvalue
                >
                auto update(const tvalue& avg_value, const tvalue& value)
                {
                        constexpr tscalar one = 1;
                        m_count += one;
                        return (avg_value * (m_count - one) + value) / m_count;
                }

                tscalar         m_count;
        };

        ///
        /// \brief running average for scalars
        ///
        template
        <
                typename tscalar,
                typename tbase = average_t<tscalar>
        >
        class average_scalar_t : private tbase
        {
        public:

                ///
                /// \brief constructor
                ///
                average_scalar_t() : m_value(0)
                {
                }

                ///
                /// \brief update the running average with a new value
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
        /// \brief running average for Eigen vectors
        ///
        template
        <
                typename tvector,
                typename tscalar = typename tvector::Scalar,
                typename tbase = average_t<tscalar>
        >
        class average_vector_t : private tbase
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
                        :       m_value(tvector::Zero(dimensions))
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

