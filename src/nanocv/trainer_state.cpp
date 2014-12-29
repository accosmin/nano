#include "trainer_state.h"
#include <fstream>

namespace ncv
{
        trainer_state_t::trainer_state_t(
                        scalar_t tvalue,
                        scalar_t terror,
                        scalar_t vvalue,
                        scalar_t verror)
                :       m_tvalue(tvalue),
                        m_terror(terror),
                        m_vvalue(vvalue),
                        m_verror(verror)
        {
        }

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
}
	
