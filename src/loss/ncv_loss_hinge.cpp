#include "ncv_loss_hinge.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        scalar_t hinge_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                scalar_t value = 0.0l;
                for (int o = 0; o < targets.rows(); o ++)
                {
                        const scalar_t edge = targets[o] * scores[o];
                        value += std::max(1.0l - edge, 0.0l);
                }
                return value;
        }

        //-------------------------------------------------------------------------------------------------
        
        scalar_t hinge_loss_t::vgrad(const vector_t& targets, const vector_t& scores, vector_t& grads) const
        {
                scalar_t value = 0.0l;
                for (int o = 0; o < targets.rows(); o ++)
                {
                        const scalar_t edge = targets[o] * scores[o];
                        value += std::max(1.0l - edge, 0.0l);
                        grads[o] = edge > 1.0l ? 0.0l : - targets[o];
                }
                return value;
        }

        //-------------------------------------------------------------------------------------------------
}
