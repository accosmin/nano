#include "avg_criterion.h"
#include <cassert>

namespace ncv
{        
        avg_criterion_t::avg_criterion_t(const string_t& configuration)
                :       criterion_t(configuration),
                        m_value(0.0),
                        m_error(0.0),
                        m_count(0)
        {
        }

        void avg_criterion_t::reset()
        {
                m_value = 0.0;
                m_error = 0.0;
                m_count = 0;
                m_vgrad.resize(psize());
                m_vgrad.setZero();
        }

        criterion_t& avg_criterion_t::operator+=(const criterion_t& other)
        {
                const avg_criterion_t* vother = dynamic_cast<const avg_criterion_t*>(&other);
                assert(vother != nullptr);
                
                m_value += vother->m_value;
                m_vgrad += vother->m_vgrad;
                m_error += vother->m_error;
                m_count += vother->m_count;
                return *this;
        }

        void avg_criterion_t::accumulate(scalar_t value, scalar_t error)
        {
                m_value += value;
                m_error += error;
                m_count ++;
        }

        void avg_criterion_t::accumulate(const vector_t& vgrad, scalar_t value, scalar_t error)
        {
                m_value += value;
                m_error += error;
                m_count ++;

                m_vgrad += vgrad;
        }
        
        scalar_t avg_criterion_t::value() const
        {
                assert(count() > 0);

                return m_value / count();
        }

        scalar_t avg_criterion_t::error() const
        {
                assert(count() > 0);

                return m_error / count();
        }

        vector_t avg_criterion_t::vgrad() const
        {
                assert(count() > 0);

                return m_vgrad / count();
        }

        size_t avg_criterion_t::count() const
        {
                return m_count;
        }

        bool avg_criterion_t::can_regularize() const
        {
                return false;
        }
}
	
