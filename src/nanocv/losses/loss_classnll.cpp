#include "loss_classnll.h"
#include <cassert>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        classnll_loss_t::classnll_loss_t()
                :       loss_t(string_t(), "class negative log-likelihood loss")
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t classnll_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return  std::log(scores.array().exp().sum()) - 
                        0.5 * (targets.array() + 1.0).matrix().dot(scores);
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        
        vector_t classnll_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return  scores.array().exp().matrix() / scores.array().exp().sum() -
                        0.5 * (targets.array() + 1.0).matrix();
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
