#pragma once

#include <cmath>
#include <limits>
#include <ostream>
#include "core/timer.h"
#include "solvers/state.h"

namespace nano
{
        ///
        /// \brief dataset measurement (e.g. after a training epoch).
        ///
        struct trainer_measurement_t
        {
                ///
                /// \brief check validity of measurements (detect divergence)
                ///
                operator bool() const
                {
                        return std::isfinite(m_value) && std::isfinite(m_error);
                }

                // attributes
                scalar_t        m_value{std::numeric_limits<scalar_t>::max()};  ///< average loss value
                scalar_t        m_error{std::numeric_limits<scalar_t>::max()};  ///< average error
        };

        inline std::ostream& operator<<(std::ostream& os, const trainer_measurement_t& measure)
        {
                return os << measure.m_value << "|" << measure.m_error;
        }

        inline bool operator<(const trainer_measurement_t& one, const trainer_measurement_t& two)
        {
                return  ((one) ? one.m_error : std::numeric_limits<scalar_t>::max()) <
                        ((two) ? two.m_error : std::numeric_limits<scalar_t>::max());
        }

        ///
        /// \brief training state after a training epoch.
        ///
        struct trainer_state_t
        {
                ///
                /// \brief check validity
                ///
                operator bool() const
                {
                        return m_train && m_valid && m_test;
                }

                // attributes
                milliseconds_t          m_milis{0};     ///< (cumulated) elapsed time since the optimization started
                size_t                  m_epoch{0};     ///<
                scalar_t                m_xnorm{0};     ///< L2-norm of the parameters
                scalar_t                m_gnorm{0};     ///< L2-norm of the parameters' gradient
                trainer_measurement_t   m_train;        ///< measurement on the training dataset
                trainer_measurement_t   m_valid;        ///< measurement on the validation dataset
                trainer_measurement_t   m_test;         ///< measurement on the test dataset
        };

        using trainer_states_t = std::vector<trainer_state_t>;

        inline bool operator<(const trainer_state_t& one, const trainer_state_t& two)
        {
                // compare (aka tune) on the validation dataset!
                return one.m_valid < two.m_valid;
        }

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

                trainer_result_t() = default;

                ///
                /// \brief constructor
                ///
                explicit trainer_result_t(string_t config);

                ///
                /// \brief update the current/optimum state with a possible better state
                ///
                trainer_status update(const solver_state_t&, const trainer_state_t&, const size_t patience);

                ///
                /// \brief check if a valid result
                ///
                operator bool() const
                {
                        return !m_history.empty() && m_opt_params.size() > 0;
                }

                ///
                /// \brief computes the convergence speed (aka. loss decrease ratio per unit time)
                ///
                scalar_t convergence_speed() const;

                ///
                /// \brief save the training history as csv
                ///
                bool save(const string_t& path) const;

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

        inline bool operator<(const trainer_result_t& result1, const trainer_result_t& result2)
        {
                return result1.optimum_state() < result2.optimum_state();
        }

        inline std::ostream& operator<<(std::ostream& os, const trainer_result_t& result)
        {
                const auto& state = result.optimum_state();
                return  os
                        << "train=" << state.m_train
                        << ",valid=" << state.m_valid
                        << ",test=" << state.m_test
                        << "," << result.config() << ",epoch=" << result.optimum_epoch()
                        << ",speed=" << (result.history().size() > 1 ? result.convergence_speed() : scalar_t(0)) << "/s";
        }
}
