#include "var_criterion.h"

namespace ncv
{        
        var_criterion_t::var_criterion_t(
                const std::string& configuration)
                :       criterion_t(configuration, "variational (EBBoost-like) regularized loss")
        {
        }

        void var_criterion_t::reset()
        {
                criterion_t::reset();
                
                m_sumg.resize(m_params.size());
                m_sumvg.resize(m_params.size());                
                
                m_sumv = 0.0;
                m_sumvv = 0.0;
                m_sumg.setZero();
                m_sumvg.setZero();
        }

        criterion_t& var_criterion_t::operator+=(const criterion_t& other)
        {
                criterion_t::operator+=(other);
                
                const var_criterion_t* vother = dynamic_cast<const var_criterion_t*>(&other);
                assert(vother != nullptr);
                
                m_sumv += vother->m_sumv;
                m_sumvv += vother->m_sumvv;
                m_sumg += vother->m_sumg;
                m_sumvg += vother->m_sumvg;
                
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
                
                m_sumv += crt_value;
                m_sumvv += crt_value * crt_value;
                m_sumg += crt_vgrad;
                m_sumvg += crt_value * crt_vgrad;
        }
        
        scalar_t var_criterion_t::value() const
        {
                return  criterion_t::value() +
                        m_lambda * (count() * m_sumvv - m_sumv * m_sumv) / (count() * count());
        }

        scalar_t var_criterion_t::error() const
        {
                return  criterion_t::error();
        }

        vector_t var_criterion_t::vgrad() const
        {
                return  criterion_t::vgrad() +
                        m_lambda * (count() * m_sumvg - 2.0 * m_sumv * m_sumg) / (count() * count());
        }
}
	
