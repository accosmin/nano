#include "ncv_loss_classnll.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        classnll_loss_t::classnll_loss_t(const string_t&)
                :       loss_t("classnll",
                               "class negative log-likelihood loss")
        {
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t classnll_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                return  std::log(scores.array().exp().sum()) - 
                        0.5 * (targets.array() + 1.0).matrix().dot(scores);
        }

        //-------------------------------------------------------------------------------------------------
        
        scalar_t classnll_loss_t::vgrad(const vector_t& targets, const vector_t& scores, vector_t& grads) const
        {
                grads = scores.array().exp().matrix() / scores.array().exp().sum() - 
                        0.5 * (targets.array() + 1.0).matrix();
                
                return value(targets, scores);
        }

        //-------------------------------------------------------------------------------------------------
}
