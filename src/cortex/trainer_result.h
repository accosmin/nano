#pragma once

#include <map>
#include "tensor.h"
#include "trainer_state.h"
#include "text/enum_string.hpp"

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
        /// \brief return code for updating the state
        ///
        enum class trainer_result_return_t
        {
                better,         ///< performance improved
                worse,          ///< performance decreased (but not critically)
                overfitting,    ///< overfitting detected (processing should stop)
                solved          ///< problem solved with arbitrary accuracy (processing should stop)
        };

        ///
        /// \brief check if the training should be stopped based on the return code
        ///
        NANOCV_PUBLIC bool is_done(const trainer_result_return_t);
        
        ///
        /// \brief track the current/optimum model state
        ///
        class NANOCV_PUBLIC trainer_result_t
        {
        public:

                ///
                /// \brief constructor
                ///
                trainer_result_t();

                ///
                /// \brief update the current/optimum state with a possible better state
                ///
                trainer_result_return_t update(const vector_t& params,
                        scalar_t tvalue, scalar_t terror_avg, scalar_t terror_var,
                        scalar_t vvalue, scalar_t verror_avg, scalar_t verror_var,
                        size_t epoch, const scalars_t& config);

                ///
                /// \brief update the current/optimum state with a possible better state
                ///
                trainer_result_return_t update(const trainer_result_t& other);

                ///
                /// \brief check if valid result
                ///
                bool valid() const
                {
                        return !m_history.empty() && m_opt_params.size() > 0;
                }

                ///
                /// \brief optimum training state
                ///
                trainer_state_t optimum_state() const;
                
                ///
                /// \brief training history for the optimum configuration
                ///
                trainer_states_t optimum_states() const;

                ///
                /// \brief optimum model parameters
                ///
                vector_t optimum_params() const;

                ///
                /// \brief optimum hyper-parameter configuration
                ///
                trainer_config_t optimum_config() const;

                ///
                /// \brief optimum epoch
                ///
                size_t optimum_epoch() const;

        private:

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
        NANOCV_PUBLIC bool operator<(const trainer_result_t& one, const trainer_result_t& other);
}

namespace text
{
        template <>
        inline std::map<ncv::trainer_result_return_t, std::string> enum_string<ncv::trainer_result_return_t>()
        {
                return
                {
                        { ncv::trainer_result_return_t::better,      "better" },
                        { ncv::trainer_result_return_t::worse,       "worse" },
                        { ncv::trainer_result_return_t::overfitting, "overfitting" },
                        { ncv::trainer_result_return_t::solved,      "solved" }
                };
        }
}

