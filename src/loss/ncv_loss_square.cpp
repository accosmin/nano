#include "ncv_loss_square.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        square_loss_t::square_loss_t(const string_t&)
                :       loss_t("square",
                               "square loss")
        {
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t square_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                return 0.5 * (targets.array() - scores.array()).abs2().sum();
        }

        //-------------------------------------------------------------------------------------------------
        
        scalar_t square_loss_t::vgrad(const vector_t& targets, const vector_t& scores, vector_t& grads) const
        {
                grads = targets - scores;
                return value(targets, scores);
        }

        //-------------------------------------------------------------------------------------------------
}
