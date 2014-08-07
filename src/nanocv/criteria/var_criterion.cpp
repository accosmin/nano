#include "var_criterion.h"

namespace ncv
{        
        var_criterion_t::var_criterion_t(
                const std::string& configuration)
                :       criterion_t(configuration, "variational (VadaBoost-like) regularized loss")
        {
        }

        void var_criterion_t::reset()
        {
                criterion_t::reset();

                m_value2 = 0.0;

                m_vgrad2.resize(m_params.size());
                m_vgrad2.setZero();
        }

        criterion_t& var_criterion_t::operator+=(const criterion_t& other)
        {
                criterion_t::operator+=(other);
                
                const var_criterion_t* vother = dynamic_cast<const var_criterion_t*>(&other);
                assert(vother != nullptr);
                
                m_value2 += vother->m_value2;
                m_vgrad2 += vother->m_vgrad2;
                
                return *this;
        }

        void var_criterion_t::cumulate(
                const vector_t& output, const vector_t& target, const loss_t& loss)
        {
                const scalar_t old_value = m_value;
                const vector_t old_vgrad = m_vgrad;
                
                criterion_t::cumulate(output, target, loss);
                
                const scalar_t crt_value = m_value - old_value;
                const vector_t crt_vgrad = m_vgrad - old_vgrad;
                
                m_value2 += crt_value * crt_value;
                m_vgrad2 += crt_value * crt_vgrad;
        }
        
        scalar_t var_criterion_t::value() const
        {
                return  criterion_t::value() +
                        m_lambda * (count() * m_value2 - m_value * m_value) / (count() * count());
        }

        scalar_t var_criterion_t::error() const
        {
                return  criterion_t::error();
        }

        vector_t var_criterion_t::vgrad() const
        {
                return  criterion_t::vgrad() +
                        2.0 * m_lambda * (count() * m_vgrad2 - m_value * m_vgrad) / (count() * count());
        }

        bool var_criterion_t::can_regularize() const
        {
                return true;
        }
}
	
