#pragma once

#include "timer.h"
#include "scalar.h"
#include "stringi.h"
#include <ostream>

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
                trainer_measurement_t();

                ///
                /// \brief constructor
                ///
                trainer_measurement_t(const scalar_t value, const scalar_t error_avg, const scalar_t error_var);

                // attributes
                scalar_t        m_value;        ///< loss value
                scalar_t        m_error_avg;    ///< error (average)
                scalar_t        m_error_var;    ///< error (variance)
        };

        inline std::ostream& operator<<(std::ostream& os, const trainer_measurement_t& measure)
        {
                return os << measure.m_value << "|" << measure.m_error_avg << "+/-" << measure.m_error_var;
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
                trainer_state_t(const milliseconds_t milis, const size_t epoch,
                                const trainer_measurement_t& train,
                                const trainer_measurement_t& valid,
                                const trainer_measurement_t& test);

                // attributes
                milliseconds_t          m_milis;        ///< (cumulated) elapsed time since the optimization started
                size_t                  m_epoch;        ///<
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

