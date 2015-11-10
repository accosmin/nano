#pragma once

namespace math
{
        ///
        /// \brief moving weighted-average for vectors
        ///
        template
        <
                typename tscalar,
                typename tvector
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
                explicit average_vector_t(tsize dimensions)
                        :       m_weights(tscalar(0)),
                                m_average(dimensions)
                {
                        m_average.setZero();
                }

                ///
                /// \brief update the running average with a new value
                ///
                void update(const tvector& value, tscalar weight)
                {
                        m_average.noalias() = (m_average * m_weights + value * weight) / (m_weights + weight);
                        m_weights = m_weights + weight;
                }

                ///
                /// \brief retrieve the current average
                ///
                const tvector& value() const
                {
                        return m_average;
                }

        private:

                tscalar         m_weights;      ///< cumulated weights
                tvector         m_average;      ///< average
        };
}

