#pragma once

#include "trainer_state.h"
#include <map>

namespace ncv
{
        ///
        /// \brief training configuration (e.g. learning rate, regularization weight)
        ///
        typedef scalars_t               trainer_config_t;
        
        ///
        /// \brief training history (configuration, optimization states)
        ///
        typedef std::map
        <
                trainer_config_t,
                trainer_states_t
        >                               trainer_history_t;
        
        ///
        /// \brief track the current/optimum model state
        ///
        struct trainer_result_t
        {
                ///
                /// \brief constructor
                ///
                trainer_result_t();

                ///
                /// \brief update the current/optimum state with a possible better state
                /// \return true is the state was improved (aka lower validation error)
                ///
                bool update(const vector_t& params,
                            scalar_t tvalue, scalar_t terror_avg, scalar_t terror_var,
                            scalar_t vvalue, scalar_t verror_avg, scalar_t verror_var,
                            size_t epoch, const scalars_t& config);
                bool update(const trainer_result_t& other);

                ///
                /// \brief check if valid result
                ///
                bool valid() const
                {
                        return !m_history.empty() && m_opt_params.size() > 0;
                }
                
                ///
                /// \brief training history for the optimum configuration
                ///
                trainer_states_t optimum_states() const;

                // attributes
                vector_t                m_opt_params;           ///< optimum model parameters
                trainer_state_t         m_opt_state;            ///< optimum training state
                trainer_config_t        m_opt_config;           ///< optimum configuration
                size_t                  m_opt_epoch;            ///< optimum epoch
                trainer_history_t       m_history;              ///< optimization history
        };

        ///
        /// \brief compare two trainer results
        ///
        inline bool operator<(const trainer_result_t& one, const trainer_result_t& other)
        {
                return one.m_opt_state < other.m_opt_state;
        }
}

