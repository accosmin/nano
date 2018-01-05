#pragma once

#include <cmath>
#include <limits>
#include <ostream>
#include "scalar.h"
#include "chrono/timer.h"

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

        ///
        /// \brief training state after a training epoch.
        ///
        struct trainer_state_t
        {
                ///
                /// \brief check validity
                ///
                operator bool() const
                {
                        return m_train && m_valid && m_test;
                }

                // attributes
                milliseconds_t          m_milis{0};     ///< (cumulated) elapsed time since the optimization started
                size_t                  m_epoch{0};     ///<
                scalar_t                m_xnorm{0};     ///< L2-norm of the parameters
                scalar_t                m_gnorm{0};     ///< L2-norm of the parameters' gradient
                trainer_measurement_t   m_train;        ///< measurement on the training dataset
                trainer_measurement_t   m_valid;        ///< measurement on the validation dataset
                trainer_measurement_t   m_test;         ///< measurement on the test dataset
        };

        using trainer_states_t = std::vector<trainer_state_t>;

        inline bool operator<(const trainer_state_t& one, const trainer_state_t& two)
        {
                // compare (aka tune) on the validation dataset!
                return one.m_valid < two.m_valid;
        }
}
