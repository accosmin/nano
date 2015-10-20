#include "avg_criterion.h"
#include <cassert>

namespace cortex
{        
        avg_criterion_t::avg_criterion_t(const string_t& configuration)
                :       criterion_t(configuration),
                        m_value(0.0)
        {
        }

        void avg_criterion_t::clear()
        {
                m_value = 0.0;
                m_vgrad.resize(psize());
                m_vgrad.setZero();
        }

        void avg_criterion_t::accumulate(scalar_t value)
        {
                m_value += value;                
        }

        void avg_criterion_t::accumulate(const vector_t& vgrad, scalar_t value)
        {
                m_value += value;
                m_vgrad += vgrad;
        }

        void avg_criterion_t::accumulate(const criterion_t& other)
        {
                const avg_criterion_t* vother = dynamic_cast<const avg_criterion_t*>(&other);
                assert(vother != nullptr);

                m_value += vother->m_value;
                m_vgrad += vother->m_vgrad;
        }
        
        scalar_t avg_criterion_t::value() const
        {
                assert(count() > 0);

                return m_value / static_cast<scalar_t>(count());
        }

        vector_t avg_criterion_t::vgrad() const
        {
                assert(count() > 0);

                return m_vgrad / static_cast<scalar_t>(count());
        }

        bool avg_criterion_t::can_regularize() const
        {
                return false;
        }
}
	
