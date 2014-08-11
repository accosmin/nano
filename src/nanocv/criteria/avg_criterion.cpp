#include "avg_criterion.h"
#include "loss.h"
#include <cassert>

namespace ncv
{        
        avg_criterion_t::avg_criterion_t(const string_t& configuration, const string_t& description)
                :       criterion_t(configuration, description),
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

        void avg_criterion_t::accumulate(
                const vector_t& output, const vector_t& target, const loss_t& loss)
        {
                assert(static_cast<size_t>(output.size()) == m_model->osize());
                assert(static_cast<size_t>(target.size()) == m_model->osize());
                
                // loss value
                m_value += loss.value(target, output);
                m_error += loss.error(target, output);
                m_count ++;
                
                // loss gradient
                switch (m_type)
                {
                case type::value:
                        break;

                case type::vgrad:
                        m_vgrad += m_model->pgrad(loss.vgrad(target, output));
                        break;
                }
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
	
