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
        }

        criterion_t& var_criterion_t::operator+=(const criterion_t& other)
        {
                return *this;
        }

        void var_criterion_t::cumulate(
                const vector_t& output, const vector_t& target, const loss_t& loss)
        {
                criterion_t::cumulate(output, target, loss);
        }
        
        scalar_t var_criterion_t::value() const
        {
                return  criterion_t::value() +
                        0.5 * m_lambda * m_params.squaredNorm();
        }

        scalar_t var_criterion_t::error() const
        {
                return  criterion_t::error();
        }

        vector_t var_criterion_t::vgrad() const
        {
                return  criterion_t::vgrad() +
                        m_lambda * m_params;
        }
}
	
