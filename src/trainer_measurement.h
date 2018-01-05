#pragma once

#include <limits>
#include <ostream>
#include "scalar.h"

namespace nano
{
        ///
        /// \brief dataset measurement (e.g. after a training epoch).
        ///
        struct trainer_measurement_t
        {
                ///
                /// \brief check validity of measurements (detect divergence)
                ///
                operator bool() const
                {
                        return std::isfinite(m_value) && std::isfinite(m_error);
                }

                // attributes
                scalar_t        m_value{std::numeric_limits<scalar_t>::max()};  ///< average loss value
                scalar_t        m_error{std::numeric_limits<scalar_t>::max()};  ///< average error
        };

        inline std::ostream& operator<<(std::ostream& os, const trainer_measurement_t& measure)
        {
                return os << measure.m_value << "|" << measure.m_error;
        }

        inline bool operator<(const trainer_measurement_t& one, const trainer_measurement_t& two)
        {
                return  ((one) ? one.m_value : std::numeric_limits<scalar_t>::max()) <
                        ((two) ? two.m_value : std::numeric_limits<scalar_t>::max());
        }
}
