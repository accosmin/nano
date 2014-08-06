#include "l2_criterion.h"

namespace ncv
{        
        l2_criterion_t::l2_criterion_t(
                const std::string& configuration)
                :       criterion_t(configuration, "L2-norm regularized loss")
        {
        }

        void l2_criterion_t::reset()
        {
                criterion_t::reset();
        }

        criterion_t& l2_criterion_t::operator+=(const criterion_t& other)
        {
                criterion_t::operator+=(other);
                return *this;
        }

        void l2_criterion_t::cumulate(
                const vector_t& output, const vector_t& target, const loss_t& loss)
        {
                criterion_t::cumulate(output, target, loss);
        }
        
        scalar_t l2_criterion_t::value() const
        {
                return  criterion_t::value() +
                        0.5 * m_lambda * m_params.squaredNorm();
        }

        scalar_t l2_criterion_t::error() const
        {
                return  criterion_t::error();
        }

        vector_t l2_criterion_t::vgrad() const
        {
                return  criterion_t::vgrad() +
                        m_lambda * m_params;
        }

        bool l2_criterion_t::can_regularize() const
        {
                return true;
        }
}
	
