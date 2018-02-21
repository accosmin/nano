#pragma once

namespace nano
{
        ///
        /// \brief running average for Eigen vectors.
        ///
        template <typename tvector, typename tscalar = typename tvector::Scalar>
        class average_t
        {
        public:
                template <typename tsize>
                explicit average_t(const tsize dimensions) :
                        m_value(std::move(tvector::Zero(dimensions)))
                {
                }

                template <typename texpression>
                void update(texpression&& value)
                {
                        m_count ++;
                        m_value.array() = (m_value.array() * (m_count - 1) + value.array()) * (1 / m_count);
                }

                const auto& value() const
                {
                        return m_value;
                }

        private:

                tscalar         m_count{0};
                tvector         m_value;        ///< running average
        };
}

