#pragma once

#include "util/arch.h"
#include "string.h"
#include "scalar.h"
#include <limits>
#include <cmath>

namespace ncv
{
        ///
        /// \brief training state
        ///
        struct NANOCV_DLL_PUBLIC trainer_state_t
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
        inline bool operator<(const trainer_state_t& one, const trainer_state_t& two)
        {
                const scalar_t v1 = std::isfinite(one.m_verror_avg) ? one.m_verror_avg : std::numeric_limits<scalar_t>::max();
                const scalar_t v2 = std::isfinite(two.m_verror_avg) ? two.m_verror_avg : std::numeric_limits<scalar_t>::max();
                return v1 < v2;
        }
        
        typedef std::vector<trainer_state_t>    trainer_states_t;

        ///
        /// \brief save optimization states to text file
        ///
        NANOCV_DLL_PUBLIC bool save(const string_t& path, const trainer_states_t& states);
}

