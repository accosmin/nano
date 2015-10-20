#include "avg_var_criterion.h"
#include <cassert>

namespace cortex
{        
        avg_var_criterion_t::avg_var_criterion_t(const string_t& configuration)
                :       avg_criterion_t(configuration),
                        m_value2(0.0)
        {
        }

        void avg_var_criterion_t::clear()
        {
                avg_criterion_t::clear();

                m_value2 = 0.0;

                m_vgrad2.resize(psize());
                m_vgrad2.setZero();
        }

        void avg_var_criterion_t::accumulate(scalar_t value)
        {
                avg_criterion_t::accumulate(value);
                
                m_value2 += value * value;
        }

        void avg_var_criterion_t::accumulate(const vector_t& vgrad, scalar_t value)
        {
                avg_criterion_t::accumulate(vgrad, value);

                m_value2 += value * value;
                m_vgrad2 += value * vgrad;
        }

        void avg_var_criterion_t::accumulate(const criterion_t& other)
        {
                avg_criterion_t::accumulate(other);

                const avg_var_criterion_t* vother = dynamic_cast<const avg_var_criterion_t*>(&other);
                assert(vother != nullptr);

                m_value2 += vother->m_value2;
                m_vgrad2 += vother->m_vgrad2;
        }

        scalar_t avg_var_criterion_t::value() const
        {
                const scalar_t count = static_cast<scalar_t>(this->count());

                return  lweight() * (avg_criterion_t::value()) +
                        rweight() * (count * m_value2 - m_value * m_value) / (count * count);
        }

        vector_t avg_var_criterion_t::vgrad() const
        {
                const scalar_t count = static_cast<scalar_t>(this->count());

                return  lweight() * (avg_criterion_t::vgrad()) +
                        rweight() * (2 * (count * m_vgrad2 - m_value * m_vgrad) / (count * count));
        }

        bool avg_var_criterion_t::can_regularize() const
        {
                return true;
        }
}
	
