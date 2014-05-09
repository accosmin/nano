#ifndef NANOCV_LOSS_CLASS_DIF_H
#define NANOCV_LOSS_CLASS_DIF_H

#include "loss.h"

namespace ncv
{
        ///
        /// \brief multi-class positive vs negative difference loss
        ///
        class classdif_loss_t : public loss_t
        {
        public:

                // constructor
                classdif_loss_t();

                // create an object clone
                virtual rloss_t clone(const string_t&) const
                {
                        return rloss_t(new classdif_loss_t);
                }

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const
                {
                        return multi_class_error(targets, scores);
                }

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const;
        };
}

#endif // NANOCV_LOSS_CLASS_DIF_H
