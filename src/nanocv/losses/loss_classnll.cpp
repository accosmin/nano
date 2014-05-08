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

                vector_t probs = scores.array().exp();
                probs.noalias() = probs / probs.sum();

                return -probs.dot(targets);
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        
        vector_t classnll_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                vector_t probs = scores.array().exp();
                probs.noalias() = probs / probs.sum();

                const scalar_t tprobs = probs.dot(targets);

                return probs.array() * (tprobs - targets.array());
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
