#include "avg_l2_criterion.h"

namespace ncv
{        
        avg_l2_criterion_t::avg_l2_criterion_t(const std::string& configuration)
                :       avg_criterion_t(configuration)
        {
        }

        void avg_l2_criterion_t::reset()
        {
                avg_criterion_t::reset();
        }

        criterion_t& avg_l2_criterion_t::operator+=(const criterion_t& other)
        {
                avg_criterion_t::operator+=(other);
                return *this;
        }

        void avg_l2_criterion_t::accumulate(scalar_t value, scalar_t error)
        {
                avg_criterion_t::accumulate(value, error);
        }

        void avg_l2_criterion_t::accumulate(const vector_t& vgrad, scalar_t value, scalar_t error)
        {
                avg_criterion_t::accumulate(vgrad, value, error);
        }
        
        scalar_t avg_l2_criterion_t::value() const
        {
                return  avg_criterion_t::value() +
                        0.5 * m_lambda * m_params.squaredNorm() / psize();
        }

        scalar_t avg_l2_criterion_t::error() const
        {
                return  avg_criterion_t::error();
        }

        vector_t avg_l2_criterion_t::vgrad() const
        {
                return  avg_criterion_t::vgrad() +
                        m_lambda * m_params / psize();
        }

        bool avg_l2_criterion_t::can_regularize() const
        {
                return true;
        }
}
	
