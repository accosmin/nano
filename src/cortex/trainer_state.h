#pragma once

#include "arch.h"
#include "scalar.h"
#include "string.h"

namespace cortex
{
        ///
        /// \brief training state
        ///
        struct NANOCV_PUBLIC trainer_state_t
        {
                ///
                /// \brief constructor
                ///
                trainer_state_t();

                ///
                /// \brief constructor
                ///
                trainer_state_t(size_t milis, size_t epoch,
                                scalar_t tvalue, scalar_t terror_avg, scalar_t terror_var,
                                scalar_t vvalue, scalar_t verror_avg, scalar_t verror_var);

                // attributes
                size_t          m_milis;        ///< (cumulated) elapsed miliseconds since the optimization started
                size_t          m_epoch;        ///<
                scalar_t        m_tvalue;       ///< train loss value
                scalar_t        m_terror_avg;   ///< train error (average)
                scalar_t        m_terror_var;   ///< train error (variance)
                scalar_t        m_vvalue;       ///< validation loss value
                scalar_t        m_verror_avg;   ///< validation error (average)
                scalar_t        m_verror_var;   ///< validation error (variance)
        };

        using trainer_states_t = std::vector<trainer_state_t>;

        ///
        /// \brief compute the average convergence speed of the training loss for a given set of states
        ///
        NANOCV_PUBLIC scalar_t convergence_speed(const trainer_states_t& states);

        ///
        /// \brief compare two training states
        ///
        NANOCV_PUBLIC bool operator<(const trainer_state_t& one, const trainer_state_t& two);

        ///
        /// \brief save optimization states to text file
        ///
        NANOCV_PUBLIC bool save(const string_t& path, const trainer_states_t& states);
}

