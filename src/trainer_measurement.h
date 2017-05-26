#pragma once

#include <limits>
#include <ostream>
#include "scalar.h"

namespace nano
{
        ///
        /// \brief training measurement usually after a training epoch.
        ///
        struct trainer_measurement_t
        {
                ///
                /// \brief constructor
                ///
                trainer_measurement_t(
                        const scalar_t value = std::numeric_limits<scalar_t>::max(),
                        const scalar_t error = std::numeric_limits<scalar_t>::max()) :
                        m_value(value),
                        m_error(error)
                {
                }

                ///
                /// \brief check validity of measurements (detect divergence)
                ///
                operator bool() const
                {
                        return std::isfinite(m_value) && std::isfinite(m_error);
                }

                // attributes
                scalar_t        m_value;        ///< average loss value
                scalar_t        m_error;        ///< average error
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
