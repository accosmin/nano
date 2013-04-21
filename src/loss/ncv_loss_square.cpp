#include "ncv_loss_square.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        scalar_t square_loss::value(const vector_t& targets, const vector_t& scores) const
        {
                return 0.5 * (targets.array() - scores.array()).abs2().sum();
        }

        //-------------------------------------------------------------------------------------------------
        
        scalar_t square_loss::vgrad(const vector_t& targets, const vector_t& scores, vector_t& grads) const
        {
                grads = targets - scores;
                return value(targets, scores);
        }

        //-------------------------------------------------------------------------------------------------
}
