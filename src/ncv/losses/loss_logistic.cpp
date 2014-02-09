#include "loss_logistic.h"
#include <cassert>

namespace ncv
{
        static const scalar_t delta = std::exp(1.0) - 1.0;

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t logistic_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                scalar_t value = 0.0;
                for (auto o = 0; o < targets.rows(); o ++)
                {
                        const scalar_t e = std::exp(- scores[o] * targets[o]);
                        value += std::log(delta + e);
                }
                return value;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        
        vector_t logistic_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                vector_t grads(targets.rows());
                for (auto o = 0; o < targets.rows(); o ++)
                {
                        const scalar_t e = std::exp(- scores[o] * targets[o]);
                        grads[o] = - targets[o] * e / (delta + e);
                }

                return grads;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
