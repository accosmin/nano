#include "loss_hinge.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        hinge_loss_t::hinge_loss_t(const string_t&)
                :       loss_t("hinge loss")
        {
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t hinge_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                scalar_t value = 0.0;
                for (int o = 0; o < targets.rows(); o ++)
                {
                        const scalar_t edge = targets[o] * scores[o];
                        value += std::max(1.0 - edge, 0.0);
                }

                return value;
        }

        //-------------------------------------------------------------------------------------------------
        
        vector_t hinge_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                vector_t grads(targets.rows());
                for (int o = 0; o < targets.rows(); o ++)
                {
                        const scalar_t edge = targets[o] * scores[o];
                        grads[o] = edge > 1.0 ? 0.0 : - targets[o];
                }

                return grads;
        }

        //-------------------------------------------------------------------------------------------------
}
