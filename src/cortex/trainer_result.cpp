#include "trainer_result.h"
#include "math/stats.hpp"
#include "math/epsilon.hpp"
#include "text/concatenate.hpp"

namespace cortex
{
        trainer_result_t::trainer_result_t()
        {
        }

        trainer_result_return_t trainer_result_t::update(const vector_t& params,
                const trainer_state_t& state, const scalars_t& config)
        {
                m_history[config].push_back(state);

                const scalar_t beste = m_opt_state.m_verror_avg;
                const scalar_t curre = state.m_verror_avg;

                const size_t max_epochs_without_improvement = 32;

                // arbitrary precision (problem solved!)
                if (curre < std::numeric_limits<scalar_t>::epsilon())
                {
                        m_opt_params = params;
                        m_opt_state = state;
                        m_opt_config = config;

                        return trainer_result_return_t::solved;
                }

                // improved performance
                else if (curre < beste)
                {
                        m_opt_params = params;
                        m_opt_state = state;
                        m_opt_config = config;

                        return trainer_result_return_t::better;
                }

                // worse performance
                else
                {
                        // not enough epochs, keep training
                        if (state.m_epoch < max_epochs_without_improvement)
                        {
                                return trainer_result_return_t::worse;
                        }

                        else
                        {
                                // last improvement not far in the past, keep training
                                if (state.m_epoch < m_opt_state.m_epoch + max_epochs_without_improvement)
                                {
                                        return trainer_result_return_t::worse;
                                }

                                // no improvement since many epochs, overfitting detected
                                else
                                {
                                        return trainer_result_return_t::overfitting;
                                }
                        }
                }
        }

        trainer_result_return_t trainer_result_t::update(const trainer_result_t& other)
        {
                if (*this < other)
                {
                        *this = other;
                        return trainer_result_return_t::better;
                }

                else
                {
                        return trainer_result_return_t::worse;
                }
        }

        trainer_state_t trainer_result_t::optimum_state() const
        {
                return m_opt_state;
        }
        
        trainer_states_t trainer_result_t::optimum_states() const
        {
                const string_t str_opt_config = text::concatenate(m_opt_config, "-");
                for (const auto& it : m_history)
                {
                        const string_t str_config = text::concatenate(it.first, "-");
                        if (str_config == str_opt_config)
                        {
                                return it.second;
                        }
                }
                
                return trainer_states_t();
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

        scalar_t trainer_result_t::optimum_speed() const
        {
                const auto op = [](const trainer_state_t& prv_state, const trainer_state_t& crt_state)
                {
                        assert(crt_state.m_tvalue >= scalar_t(0));
                        assert(prv_state.m_tvalue >= scalar_t(0));
                        assert(crt_state.m_milis >= prv_state.m_milis);

                        const scalar_t epsilon = math::epsilon0<scalar_t>();
                        const auto ratio = (epsilon + crt_state.m_tvalue) / (epsilon + prv_state.m_tvalue);
                        const auto delta = size_t(1) + crt_state.m_milis - prv_state.m_milis;

                        return (ratio * scalar_t(1000.0)) / static_cast<scalar_t>(delta);
                };

                math::stats_t<scalar_t> speeds;

                const auto states = optimum_states();
                for (size_t i = 0; i + 1 < states.size(); ++ i)
                {
                        speeds(op(states[i], states[i + 1]));
                }

                return speeds.avg();
        }

        bool operator<(const trainer_result_t& one, const trainer_result_t& other)
        {
                return one.optimum_state() < other.optimum_state();
        }

        bool is_done(const trainer_result_return_t code)
        {
                return  code == trainer_result_return_t::overfitting ||
                        code == trainer_result_return_t::solved;
        }
}
	
