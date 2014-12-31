#include "avg_l2_barrier_criterion.h"
#include "util/math.hpp"

namespace ncv
{        
        avg_l2_barrier_criterion_t::avg_l2_barrier_criterion_t(const string_t& configuration)
                :       avg_criterion_t(configuration)
        {
        }

        void avg_l2_barrier_criterion_t::reset()
        {
                avg_criterion_t::reset();
        }

        void avg_l2_barrier_criterion_t::accumulate(scalar_t value, scalar_t error)
        {
                avg_criterion_t::accumulate(value, error);
        }

        void avg_l2_barrier_criterion_t::accumulate(const vector_t& vgrad, scalar_t value, scalar_t error)
        {
                avg_criterion_t::accumulate(vgrad, value, error);
        }

        void avg_l2_barrier_criterion_t::accumulate(const criterion_t& other)
        {
                avg_criterion_t::accumulate(other);
        }
        
        scalar_t avg_l2_barrier_criterion_t::value() const
        {
                return  avg_criterion_t::value() +
                        0.25 * math::square(m_params.squaredNorm());
        }

        vector_t avg_l2_barrier_criterion_t::vgrad() const
        {
                return  avg_criterion_t::vgrad() +
                        m_params.squaredNorm() * m_params;
        }

        bool avg_l2_barrier_criterion_t::can_regularize() const
        {
                return false;
        }
}
	
