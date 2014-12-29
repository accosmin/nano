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
                                scalar_t terror = std::numeric_limits<scalar_t>::max(),
                                scalar_t vvalue = std::numeric_limits<scalar_t>::max(),
                                scalar_t verror = std::numeric_limits<scalar_t>::max());
                
                // attributes
                scalar_t                m_tvalue;       ///< train loss value
                scalar_t                m_terror;       ///< train error
                scalar_t                m_vvalue;       ///< validation loss value
                scalar_t                m_verror;       ///< validation error
        };

        ///
        /// \brief compare two training states
        ///
        inline bool operator<(const trainer_state_t& one, const trainer_state_t& another)
        {
                return one.m_verror < another.m_verror;
        }
        
        typedef std::vector
        <trainer_state_t>               trainer_states_t;

        ///
        /// \brief save optimization states to text file
        ///
        bool save(const string_t& path, const trainer_states_t& states);
}

