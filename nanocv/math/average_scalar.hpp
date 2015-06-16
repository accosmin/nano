#pragma once

namespace ncv
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
                        :       m_weights(tscalar(0)),
                                m_average(tscalar(0))
                {
                }

                ///
                /// \brief update the running average with a new value
                ///
                void update(tscalar value, tscalar weight)
                {
                        m_average = (m_average * m_weights + value * weight) / (m_weights + weight);
                        m_weights = m_weights + weight;
                }

                ///
                /// \brief retrieve the current average
                ///
                tscalar value() const
                {
                        return m_average;
                }

        private:

                tscalar         m_weights;      ///< cumulated weights
                tscalar         m_average;      ///< average
        };
}

