#include "trainer_result.h"
#include "../text.h"

namespace ncv
{
        trainer_result_t::trainer_result_t()
                :       m_opt_epoch(0)
        {
        }

        bool trainer_result_t::update(const vector_t& params,
                scalar_t tvalue, scalar_t terror_avg, scalar_t terror_var,
                scalar_t vvalue, scalar_t verror_avg, scalar_t verror_var,
                size_t epoch, const scalars_t& config)
        {
                const trainer_state_t state(tvalue, terror_avg, terror_var, vvalue, verror_avg, verror_var);
                m_history[config].push_back(state);
                
                if (state < m_opt_state)
                {
                        m_opt_params = params;
                        m_opt_state = state;
                        m_opt_epoch = epoch;
                        m_opt_config = config;

                        return true;
                }

                else
                {
                        return false;
                }
        }

        bool trainer_result_t::update(const trainer_result_t& other)
        {
                if (*this < other)
                {
                        *this = other;
                        return true;
                }

                else
                {
                        return false;
                }
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
}
	
