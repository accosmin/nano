#include "avg_l2_criterion.h"

namespace ncv
{        
        avg_l2_criterion_t::avg_l2_criterion_t(const string_t& configuration)
                :       avg_criterion_t(configuration)
        {
        }

        void avg_l2_criterion_t::clear()
        {
                avg_criterion_t::clear();
        }

        void avg_l2_criterion_t::accumulate(scalar_t value)
        {
                avg_criterion_t::accumulate(value);
        }

        void avg_l2_criterion_t::accumulate(const vector_t& vgrad, scalar_t value)
        {
                avg_criterion_t::accumulate(vgrad, value);
        }

        void avg_l2_criterion_t::accumulate(const criterion_t& other)
        {
                avg_criterion_t::accumulate(other);
        }
        
        scalar_t avg_l2_criterion_t::value() const
        {
                return  lweight() * (avg_criterion_t::value()) +
                        rweight() * (0.5 * params().squaredNorm() / psize());
        }

        vector_t avg_l2_criterion_t::vgrad() const
        {
                return  lweight() * (avg_criterion_t::vgrad()) +
                        rweight() * (params() / psize());
        }

        bool avg_l2_criterion_t::can_regularize() const
        {
                return true;
        }
}
	
