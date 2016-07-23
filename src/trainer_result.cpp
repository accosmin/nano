#include "optim/state.h"
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

        trainer_state trainer_result_t::update(const state_t& opt_state,
                const trainer_state_t& state, const trainer_config_t& config)
        {
                //
                if (!state)
                {
                        return trainer_state::diverge;
                }

                m_history[config].push_back(state);

                const auto beste = m_opt_state.m_valid.m_error_avg;
                const auto curre = state.m_valid.m_error_avg;

                const auto updater = [&] ()
                {
                        m_opt_params = opt_state.x;
                        m_opt_state = state;
                        m_opt_config = config;
                };

                const size_t max_epochs_without_improvement = 32;

                // optimization finished successfully
                if (opt_state.m_status == state_t::status::converged)
                {
                        if (curre < beste)
                        {
                                updater();
                        }

                        return trainer_state::solved;
                }

                // optimization failed
                else if (opt_state.m_status == state_t::status::failed)
                {
                        return trainer_state::failed;
                }

                // improved performance
                else if (curre < beste)
                {
                        updater();

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
                        return  code == trainer_state::diverge ||
                                code == trainer_state::overfit ||
                                code == trainer_state::solved ||
                                code == trainer_state::failed;

                case trainer_policy::all_epochs:
                default:
                        return  code == trainer_state::diverge ||
                                code == trainer_state::failed;
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

