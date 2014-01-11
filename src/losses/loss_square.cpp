#include "loss_square.h"
#include <cassert>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t square_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return 0.5 * (targets.array() - scores.array()).abs2().sum();
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        
        vector_t square_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return scores - targets;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
