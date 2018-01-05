#pragma once

#include "arch.h"
#include "tensor.h"
#include "text/cast.h"
#include "trainer_state.h"

namespace nano
{
        class solver_state_t;

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
        /// \brief track the current/optimum model state.
        ///
        class NANO_PUBLIC trainer_result_t
        {
        public:
                ///
                /// \brief update the current/optimum state with a possible better state
                ///
                trainer_status update(const solver_state_t&, const trainer_state_t&,
                        const string_t& config, const size_t patience);

                ///
                /// \brief update the current/optimum state with a possible better state
                ///
                trainer_status update(const trainer_result_t& other);

                ///
                /// \brief check if a valid result
                ///
                operator bool() const
                {
                        return !m_history.empty() && m_opt_params.size() > 0;
                }

                ///
                /// \brief optimum training state
                ///
                const auto& optimum_state() const { return m_opt_state; }

                ///
                /// \brief optimum model parameters
                ///
                const auto& optimum_params() const { return m_opt_params; }

                ///
                /// \brief optimum epoch
                ///
                size_t optimum_epoch() const { return optimum_state().m_epoch; }

                ///
                /// \brief training/optimization configuration
                ///
                const auto& config() const { return m_config; }

                ///
                /// \brief optimization history
                ///
                const auto& history() const { return m_history; }

        private:

                // attributes
                vector_t                m_opt_params;           ///< optimum model parameters
                trainer_state_t         m_opt_state;            ///< optimum training state
                string_t                m_config;               ///< optimization configuration
                trainer_states_t        m_history;              ///< optimization history
        };

        ///
        /// \brief streaming training results
        ///
        NANO_PUBLIC std::ostream& operator<<(std::ostream&, const trainer_result_t&);
}
