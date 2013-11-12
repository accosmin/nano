#include "loss_logistic.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t logistic_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                scalar_t value = 0.0;
                for (auto o = 0; o < targets.rows(); o ++)
                {
                        const scalar_t e = exp(- scores[o] * targets[o]);
                        value += log(1.0 + e);
                }
                return value;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        
        vector_t logistic_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                vector_t grads(targets.rows());
                for (auto o = 0; o < targets.rows(); o ++)
                {
                        const scalar_t e = exp(- scores[o] * targets[o]);
                        grads[o] = - targets[o] * e / (1.0 + e);
                }

                return grads;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
