#pragma once

#include "types.h"

namespace ncv
{
        ///
        /// \brief training state
        ///
        struct trainer_state_t
        {
                ///
                /// \brief constructor
                ///
                trainer_state_t(scalar_t tvalue = std::numeric_limits<scalar_t>::max(),
                                scalar_t terror_avg = std::numeric_limits<scalar_t>::max(),
                                scalar_t terror_var = std::numeric_limits<scalar_t>::max(),
                                scalar_t vvalue = std::numeric_limits<scalar_t>::max(),
                                scalar_t verror_avg = std::numeric_limits<scalar_t>::max(),
                                scalar_t verror_var = std::numeric_limits<scalar_t>::max());
                
                // attributes
                scalar_t                m_tvalue;       ///< train loss value
                scalar_t                m_terror_avg;   ///< train error (average)
                scalar_t                m_terror_var;   ///< train error (variance)
                scalar_t                m_vvalue;       ///< validation loss value
                scalar_t                m_verror_avg;   ///< validation error (average)
                scalar_t                m_verror_var;   ///< validation error (variance)
        };

        ///
        /// \brief compare two training states
        ///
        inline bool operator<(const trainer_state_t& one, const trainer_state_t& another)
        {
                return one.m_verror_avg < another.m_verror_avg;
        }
        
        typedef std::vector
        <trainer_state_t>               trainer_states_t;

        ///
        /// \brief save optimization states to text file
        ///
        bool save(const string_t& path, const trainer_states_t& states);
}

