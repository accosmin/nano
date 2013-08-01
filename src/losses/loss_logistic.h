#ifndef NANOCV_LOSS_LOGISTIC_H
#define NANOCV_LOSS_LOGISTIC_H

#include "loss.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////
        // logistic loss.
        ////////////////////////////////////////////////////////////////////////////////
        
        class logistic_loss_t : public loss_t
        {
        public:

                // constructor
                logistic_loss_t(const string_t& /*params*/ = string_t()) {}

                NCV_MAKE_CLONABLE(logistic_loss_t, loss_t, "logistic loss")

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const
                {
                        return eclass_error(targets, scores);
                }

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const;
        };
}

#endif // NANOCV_LOSS_LOGISTIC_H
