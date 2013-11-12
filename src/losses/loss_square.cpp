#include "loss_square.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t square_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                return 0.5 * (targets.array() - scores.array()).abs2().sum();
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        
        vector_t square_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                return scores - targets;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
