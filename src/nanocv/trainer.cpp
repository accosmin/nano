#include "trainer.h"
#include <fstream>

namespace ncv
{
        bool save(const string_t& path, const trainer_states_t& states)
        {
                std::ofstream ofs(path.c_str(), std::ofstream::out);
                if (!ofs.is_open())
                {
                        return false;
                }
                
                const string_t delim = "\t";              
                
                // header
                ofs 
                << text::resize("train-loss", 16) << delim
                << text::resize("train-error", 16) << delim
                << text::resize("valid-loss", 16) << delim
                << text::resize("valid-error", 16) << delim << "\n";

                // optimization states
                for (const trainer_state_t& state : states)
                {
                        ofs 
                        << text::resize(text::to_string(state.m_tvalue), 16) << delim
                        << text::resize(text::to_string(state.m_terror), 16) << delim
                        << text::resize(text::to_string(state.m_vvalue), 16) << delim
                        << text::resize(text::to_string(state.m_verror), 16) << delim << "\n";
                }

                return ofs.good();
        }

        trainer_result_t::trainer_result_t()
                :       m_opt_epoch(0)
        {
        }

        bool trainer_result_t::update(const vector_t& params,
                    scalar_t tvalue, scalar_t terror,
                    scalar_t vvalue, scalar_t verror,
                    size_t epoch, const scalars_t& config)
        {
                const trainer_state_t state(tvalue, terror, vvalue, verror);
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
        
        trainer_data_t::trainer_data_t(const task_t& task,
                       const sampler_t& tsampler,
                       const sampler_t& vsampler,
                       const loss_t& loss,
                       const vector_t& x0,
                       accumulator_t& lacc,
                       accumulator_t& gacc)
                :       m_task(task),
                        m_tsampler(tsampler),
                        m_vsampler(vsampler),
                        m_loss(loss),
                        m_x0(x0),
                        m_lacc(lacc),
                        m_gacc(gacc)
        {
        }
}
	
