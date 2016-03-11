#pragma once

#include "tensor.h"
#include "trainer_state.h"
#include "text/enum_string.hpp"

namespace nano
{
        ///
        /// \brief training configuration (e.g. {name, value:learning rate, regularization weight}+)
        ///
        using trainer_config_t = std::vector<std::pair<const char*, scalar_t>>;

        NANO_PUBLIC trainer_config_t append(const trainer_config_t& config, const char* const name, const scalar_t value);

        class logger_t;
        NANO_PUBLIC logger_t& operator<<(logger_t& logger, const trainer_config_t& config);

        ///
        /// \brief training history (configuration, optimization states)
        ///
        using trainer_history_t = std::map
        <
                trainer_config_t,
                trainer_states_t
        >;

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
        NANO_PUBLIC bool is_done(const trainer_result_return_t);

        ///
        /// \brief track the current/optimum model state
        ///
        class NANO_PUBLIC trainer_result_t
        {
        public:

                ///
                /// \brief constructor
                ///
                trainer_result_t();

                ///
                /// \brief update the current/optimum state with a possible better state
                ///
                trainer_result_return_t update(const vector_t& params, const trainer_state_t&, const trainer_config_t&);

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
                trainer_history_t       m_history;              ///< optimization history
        };

        ///
        /// \brief compare two trainer results
        ///
        NANO_PUBLIC bool operator<(const trainer_result_t& one, const trainer_result_t& other);
}

namespace nano
{
        template <>
        inline std::map<nano::trainer_result_return_t, std::string> enum_string<nano::trainer_result_return_t>()
        {
                return
                {
                        { nano::trainer_result_return_t::better,      "better" },
                        { nano::trainer_result_return_t::worse,       "worse" },
                        { nano::trainer_result_return_t::overfitting, "overfitting" },
                        { nano::trainer_result_return_t::solved,      "solved" }
                };
        }
}

