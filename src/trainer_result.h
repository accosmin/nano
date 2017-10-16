#pragma once

#include <map>
#include "tensor.h"
#include "trainer_state.h"
#include "text/enum_string.h"

namespace nano
{
        struct function_state_t;

        ///
        /// \brief training history (configuration, optimization states)
        ///
        using trainer_history_t = std::map<string_t, trainer_states_t>;

        ///
        /// \brief
        ///
        enum class trainer_status
        {
                failed,         ///< optimization failed
                better,         ///< performance improved
                worse,          ///< performance decreased (but not critically)
                overfit,        ///< overfitting detected (processing should stop)
                diverge,        ///< divergence detected aka Nan/Inf (processing should stop)
                solved          ///< problem solved with arbitrary accuracy (processing should stop)
        };

        template <>
        inline enum_map_t<trainer_status> enum_string<trainer_status>()
        {
                return
                {
                        { trainer_status::failed,          "*failed" },
                        { trainer_status::better,          "+better" },
                        { trainer_status::worse,           "--worse" },
                        { trainer_status::overfit,         "overfit" },
                        { trainer_status::diverge,         "diverge" },
                        { trainer_status::solved,          "!solved" }
                };
        }

        ///
        /// \brief check if the training should be stopped
        ///
        NANO_PUBLIC bool is_done(const trainer_status);

        ///
        /// \brief track the current/optimum model state
        ///
        struct NANO_PUBLIC trainer_result_t
        {
                ///
                /// \brief update the current/optimum state with a possible better state
                ///
                trainer_status update(const function_state_t&, const trainer_state_t&,
                        const string_t& config, const size_t patience);

                ///
                /// \brief update the current/optimum state with a possible better state
                ///
                trainer_status update(const trainer_result_t& other);

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
                string_t optimum_config() const;

                ///
                /// \brief optimum epoch
                ///
                size_t optimum_epoch() const;

        private:

                // attributes
                vector_t                m_opt_params;           ///< optimum model parameters
                trainer_state_t         m_opt_state;            ///< optimum training state
                string_t                m_opt_config;           ///< optimum configuration
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

