#pragma once

#include <cmath>
#include <limits>
#include <ostream>
#include "scalar.h"
#include "core/cast.h"
#include "core/timer.h"

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
                int                     m_epoch{0};     ///<
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
        /// \brief track the current/optimum model state.
        ///
        class NANO_PUBLIC trainer_result_t
        {
        public:

                enum class status
                {
                        better,         ///< performance improved
                        worse,          ///< performance decreased (but not critically)
                        overfit,        ///< overfitting detected (processing should stop)
                        diverge,        ///< divergence detected aka Nan/Inf (processing should stop)
                };

                ///
                /// \brief default constructor
                ///
                trainer_result_t() = default;

                ///
                /// \brief constructor
                ///
                explicit trainer_result_t(string_t config);

                ///
                /// \brief update the current/optimum state with a possible better state
                ///
                trainer_result_t::status update(const trainer_state_t&, const int patience);

                ///
                /// \brief check if a valid result
                ///
                operator bool() const
                {
                        return !m_history.empty();
                }

                ///
                /// \brief check if the training should be stopped
                ///
                bool is_done() const
                {
                        return  m_status == trainer_result_t::status::diverge ||
                                m_status == trainer_result_t::status::overfit;
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
                /// \brief access functions
                ///
                auto get_status() const { return m_status; }
                const auto& config() const { return m_config; }
                const auto& history() const { return m_history; }
                const auto& optimum() const { return m_optimum; }

        private:

                // attributes
                string_t                m_config;                       ///<
                status                  m_status{status::better};       ///<
                trainer_state_t         m_optimum;                      ///< optimum state
                trainer_states_t        m_history;                      ///< optimization history
        };

        template <>
        inline enum_map_t<trainer_result_t::status> enum_string<trainer_result_t::status>()
        {
                return
                {
                        { trainer_result_t::status::better,          "+better" },
                        { trainer_result_t::status::worse,           "--worse" },
                        { trainer_result_t::status::overfit,         "overfit" },
                        { trainer_result_t::status::diverge,         "diverge" }
                };
        }

        inline std::ostream& operator<<(std::ostream& os, const trainer_result_t::status status)
        {
                return os << to_string(status);
        }

        inline bool operator<(const trainer_result_t& result1, const trainer_result_t& result2)
        {
                // compare (aka tune) on the validation dataset!
                return result1.optimum() < result2.optimum();
        }

        inline std::ostream& operator<<(std::ostream& os, const trainer_result_t& result)
        {
                const auto& state = result.optimum();
                return  os
                        << "tr=" << state.m_train
                        << ",vd=" << state.m_valid << "[" << result.get_status() << "]"
                        << ",te=" << state.m_test
                        << "," << result.config() << ",epoch=" << state.m_epoch
                        << ",speed=" << (result.history().size() > 1 ? result.convergence_speed() : scalar_t(0)) << "/s";
        }
}
