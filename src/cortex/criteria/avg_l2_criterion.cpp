#include "avg_l2_criterion.h"

namespace cortex
{
        avg_l2_criterion_t::avg_l2_criterion_t(const string_t& configuration)
                :       avg_criterion_t(configuration)
        {
        }

        scalar_t avg_l2_criterion_t::value() const
        {
                return  lweight() * (avg_criterion_t::value()) +
                        rweight() * (0.5 * params().squaredNorm() / static_cast<scalar_t>(psize()));
        }

        vector_t avg_l2_criterion_t::vgrad() const
        {
                return  lweight() * (avg_criterion_t::vgrad()) +
                        rweight() * (params() / static_cast<scalar_t>(psize()));
        }

        bool avg_l2_criterion_t::can_regularize() const
        {
                return true;
        }
}

