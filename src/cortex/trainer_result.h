#pragma once

#include "tensor.h"
#include "trainer_state.h"
#include "text/enum_string.hpp"

namespace nano
{
        struct state_t;

        ///
        /// \brief training configuration (e.g. {name, value:learning rate, regularization weight}+)
        ///
        using trainer_config_t = std::vector<std::pair<const char*, scalar_t>>;

        ///
        /// \brief append hyper-parameter to configuration
        ///
        NANO_PUBLIC trainer_config_t append(const trainer_config_t&, const char* const name, const scalar_t value);

        ///
        /// \brief streaming training configurations
        ///
        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const trainer_config_t&);

        ///
        /// \brief training history (configuration, optimization states)
        ///
        using trainer_history_t = std::map
        <
                trainer_config_t,
                trainer_states_t
        >;

        ///
        /// \brief
        ///
        enum class trainer_state
        {
                failed,         ///< optimization failed
                better,         ///< performance improved
                worse,          ///< performance decreased (but not critically)
                overfit,        ///< overfitting detected (processing should stop)
                solved          ///< problem solved with arbitrary accuracy (processing should stop)
        };

        ///
        /// \brief
        ///
        enum class trainer_policy
        {
                stop_early,     ///<
                all_epochs      ///< consume all epochs
        };

        ///
        /// \brief check if the training should be stopped
        ///
        NANO_PUBLIC bool is_done(const trainer_state, const trainer_policy);

        ///
        /// \brief track the current/optimum model state
        ///
        class NANO_PUBLIC trainer_result_t
        {
        public:

                ///
                /// \brief update the current/optimum state with a possible better state
                ///
                trainer_state update(const state_t&, const trainer_state_t&, const trainer_config_t&);

                ///
                /// \brief update the current/optimum state with a possible better state
                ///
                trainer_state update(const trainer_result_t& other);

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

        ///
        /// \brief streaming training results
        ///
        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const trainer_result_t&);
}

namespace nano
{
        template <>
        inline std::map<nano::trainer_state, std::string> enum_string<nano::trainer_state>()
        {
                return
                {
                        { nano::trainer_state::failed,          "*failed" },
                        { nano::trainer_state::better,          "+better" },
                        { nano::trainer_state::worse,           "--worse" },
                        { nano::trainer_state::overfit,         "overfit" },
                        { nano::trainer_state::solved,          "!solved" }
                };
        }

        template <>
        inline std::map<nano::trainer_policy, std::string> enum_string<nano::trainer_policy>()
        {
                return
                {
                        { nano::trainer_policy::stop_early,     "stop_early" },
                        { nano::trainer_policy::all_epochs,     "all_epochs" }
                };
        }
}

