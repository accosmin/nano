#include "trainer_result.h"
#include "nanocv/text.h"

namespace ncv
{
        trainer_result_t::trainer_result_t()
                :       m_opt_epoch(0)
        {
        }

        trainer_result_return_t trainer_result_t::update(const vector_t& params,
                scalar_t tvalue, scalar_t terror_avg, scalar_t terror_var,
                scalar_t vvalue, scalar_t verror_avg, scalar_t verror_var,
                size_t epoch, const scalars_t& config)
        {
                const trainer_state_t state(tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var);
                m_history[config].push_back(state);

                const scalar_t beste = m_opt_state.m_verror_avg;
                const scalar_t curre = verror_avg;

                const size_t max_epochs_without_improvement = 16;

                // arbitrary precision (problem solved!)
                if (curre < std::numeric_limits<scalar_t>::epsilon())
                {
                        m_opt_params = params;
                        m_opt_state = state;
                        m_opt_epoch = epoch;
                        m_opt_config = config;

                        return trainer_result_return_t::solved;
                }

                // improved performance
                else if (curre < beste)
                {
                        m_opt_params = params;
                        m_opt_state = state;
                        m_opt_epoch = epoch;
                        m_opt_config = config;

                        return trainer_result_return_t::better;
                }

                // worse performance
                else
                {
                        // not enough epochs, keep training
                        if (epoch < max_epochs_without_improvement)
                        {
                                return trainer_result_return_t::worse;
                        }

                        else
                        {
                                // last improvement not far in the past, keep training
                                if (epoch < m_opt_epoch + max_epochs_without_improvement)
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
                return m_opt_epoch;
        }

        bool operator<(const trainer_result_t& one, const trainer_result_t& other)
        {
                return one.optimum_state() < other.optimum_state();
        }
}
	
