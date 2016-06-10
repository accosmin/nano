#include "trainer_result.h"

namespace nano
{
        trainer_config_t append(const trainer_config_t& config, const char* const name, const scalar_t value)
        {
                auto ret = config;
                ret.emplace_back(name, value);
                return ret;
        }

        std::ostream& operator<<(std::ostream& os, const trainer_config_t& config)
        {
                for (size_t i = 0; i < config.size(); ++ i)
                {
                        const auto& param = config[i];
                        os << param.first << "=" << param.second << ((i + 1 == config.size()) ? "" : ",");
                }
                return os;
        }

        trainer_state trainer_result_t::update(const vector_t& params,
                const trainer_state_t& state, const trainer_config_t& config)
        {
                m_history[config].push_back(state);

                const scalar_t beste = m_opt_state.m_valid.m_error_avg;
                const scalar_t curre = state.m_valid.m_error_avg;

                const size_t max_epochs_without_improvement = 32;

                // arbitrary precision (problem solved!)
                if (curre < std::numeric_limits<scalar_t>::epsilon())
                {
                        m_opt_params = params;
                        m_opt_state = state;
                        m_opt_config = config;

                        return trainer_state::solved;
                }

                // improved performance
                else if (curre < beste)
                {
                        m_opt_params = params;
                        m_opt_state = state;
                        m_opt_config = config;

                        return trainer_state::better;
                }

                // worse performance
                else
                {
                        // not enough epochs, keep training
                        if (state.m_epoch < max_epochs_without_improvement)
                        {
                                return trainer_state::worse;
                        }

                        else
                        {
                                // last improvement not far in the past, keep training
                                if (state.m_epoch < m_opt_state.m_epoch + max_epochs_without_improvement)
                                {
                                        return trainer_state::worse;
                                }

                                // no improvement since many epochs, overfitting detected
                                else
                                {
                                        return trainer_state::overfit;
                                }
                        }
                }
        }

        trainer_state trainer_result_t::update(const trainer_result_t& other)
        {
                if (*this < other)
                {
                        *this = other;
                        return trainer_state::better;
                }

                else
                {
                        return trainer_state::worse;
                }
        }

        trainer_state_t trainer_result_t::optimum_state() const
        {
                return m_opt_state;
        }

        trainer_states_t trainer_result_t::optimum_states() const
        {
                const auto it = m_history.find(m_opt_config);
                return (it == m_history.end()) ? trainer_states_t() : it->second;
        }

        vector_t trainer_result_t::optimum_params() const
        {
                return m_opt_params;
        }

        trainer_config_t trainer_result_t::optimum_config() const
        {
                return m_opt_config;
        }

        size_t trainer_result_t::optimum_epoch() const
        {
                return optimum_state().m_epoch;
        }

        bool operator<(const trainer_result_t& one, const trainer_result_t& other)
        {
                return one.optimum_state() < other.optimum_state();
        }

        bool is_done(const trainer_state code, const trainer_policy policy)
        {
                switch (policy)
                {
                case trainer_policy::stop_early:
                        return  code == trainer_state::overfit ||
                                code == trainer_state::solved;

                case trainer_policy::all_epochs:
                default:
                        return false;
                }
        }

        std::ostream& operator<<(std::ostream& os, const trainer_result_t& result)
        {
                const auto state = result.optimum_state();

                return os << "train=" << state.m_train
                          << ", valid=" << state.m_valid
                          << ", test=" << state.m_test
                          << ", " << result.optimum_config() << ",epoch=" << result.optimum_epoch()
                          << ", speed=" << convergence_speed(result.optimum_states()) << "/s";
        }
}

