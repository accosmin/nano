#include "ncv_loss_logistic.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        scalar_t logistic_loss::value(const vector_t& targets, const vector_t& scores) const
        {
                scalar_t value = 0.0;
                for (int o = 0; o < targets.rows(); o ++)
                {
                        const scalar_t e = exp(- scores[o] * targets[o]);
                        value += log(1.0 + e);
                }
                return value;
        }

        //-------------------------------------------------------------------------------------------------
        
        scalar_t logistic_loss::vgrad(const vector_t& targets, const vector_t& scores, vector_t& grads) const
        {
                scalar_t value = 0.0;
                for (int o = 0; o < targets.rows(); o ++)
                {
                        const scalar_t e = exp(- scores[o] * targets[o]);
                        value += log(1.0 + e);
                        grads[o] = - targets[o] * e / (1.0 + e);
                }
                return value;
        }

        //-------------------------------------------------------------------------------------------------
}
