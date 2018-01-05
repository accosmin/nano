#pragma once

#include "arch.h"
#include "stringi.h"
#include "chrono/timer.h"
#include "trainer_measurement.h"

namespace nano
{
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

        ///
        /// \brief compute the average convergence speed of the training loss for a given set of states
        ///
        NANO_PUBLIC scalar_t convergence_speed(const trainer_states_t& states);

        ///
        /// \brief save optimization states to text file
        ///
        NANO_PUBLIC bool save(const string_t& path, const trainer_states_t& states);
}
