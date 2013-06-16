#ifndef NANOCV_LOSS_CLASS_NLL_H
#define NANOCV_LOSS_CLASS_NLL_H

#include "loss.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // multi-class negative log-likelihood loss.
        ////////////////////////////////////////////////////////////////////////////////
        
        class classnll_loss_t : public loss_t
        {
        public:

                // constructor
                classnll_loss_t(const string_t& params = string_t());

                // create an object clone
                virtual rloss_t clone(const string_t& params) const
                {
                        return rloss_t(new classnll_loss_t(params));
                }

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const
                {
                        return mclass_error(targets, scores);
                }

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const;
        };
}

#endif // NANOCV_LOSS_CLASS_NLL_H
