#pragma once

#include "arch.h"
#include "timer.h"
#include "scalar.h"
#include "stringi.h"
#include <ostream>

namespace nano
{
        ///
        /// \brief training measurement usually after a training epoch.
        ///
        struct NANO_PUBLIC trainer_measurement_t
        {
                ///
                /// \brief constructor
                ///
                trainer_measurement_t();

                ///
                /// \brief constructor
                ///
                trainer_measurement_t(const scalar_t value, const scalar_t error);

                ///
                /// \brief check validity of measurements (detect divergence)
                ///
                operator bool() const;

                // attributes
                scalar_t        m_value;        ///< average loss value
                scalar_t        m_error;        ///< average error
        };

        ///
        /// \brief streaming training measurement
        ///
        inline std::ostream& operator<<(std::ostream& os, const trainer_measurement_t& measure)
        {
                return os << measure.m_value << "|" << measure.m_error;
        }

        ///
        /// \brief compare two training measurements
        ///
        NANO_PUBLIC bool operator<(const trainer_measurement_t& one, const trainer_measurement_t& two);

        ///
        /// \brief training state
        ///
        struct NANO_PUBLIC trainer_state_t
        {
                ///
                /// \brief constructor
                ///
                trainer_state_t();

                ///
                /// \brief constructor
                ///
                trainer_state_t(const milliseconds_t, const size_t epoch, const scalar_t xnorm, const scalar_t gnorm,
                                const trainer_measurement_t& train,
                                const trainer_measurement_t& valid,
                                const trainer_measurement_t& test);

                ///
                /// \brief check validity
                ///
                operator bool() const
                {
                        return m_train && m_valid && m_test;
                }

                // attributes
                milliseconds_t          m_milis;        ///< (cumulated) elapsed time since the optimization started
                size_t                  m_epoch;        ///<
                scalar_t                m_xnorm;        ///< L2-norm of the parameters
                scalar_t                m_gnorm;        ///< L2-norm of the parameters' gradient
                trainer_measurement_t   m_train;        ///< measurement on the training dataset
                trainer_measurement_t   m_valid;        ///< measurement on the validation dataset
                trainer_measurement_t   m_test;         ///< measurement on the test dataset
        };

        using trainer_states_t = std::vector<trainer_state_t>;

        ///
        /// \brief compute the average convergence speed of the training loss for a given set of states
        ///
        NANO_PUBLIC scalar_t convergence_speed(const trainer_states_t& states);

        ///
        /// \brief compare two training states
        ///
        NANO_PUBLIC bool operator<(const trainer_state_t& one, const trainer_state_t& two);

        ///
        /// \brief save optimization states to text file
        ///
        NANO_PUBLIC bool save(const string_t& path, const trainer_states_t& states);
}

