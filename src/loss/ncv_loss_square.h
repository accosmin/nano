#ifndef NANOCV_LOSS_SQUARE_H
#define NANOCV_LOSS_SQUARE_H

#include "ncv_loss.h"

namespace ncv
{
	////////////////////////////////////////////////////////////////////////////////
        // square loss.
	////////////////////////////////////////////////////////////////////////////////
	
        class square_loss : public loss
	{
	public:
                
                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const
                {
                        return l1_error(targets, scores);
                }
                
                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const;
                virtual scalar_t vgrad(const vector_t& targets, const vector_t& scores, vector_t& grads) const;

                // create an object clone
                virtual rloss clone(const string_t& /*params*/) const
                {
                        return rloss(new square_loss(*this));
                }

                // describe the object
                virtual const char* name() const { return "square"; }
                virtual const char* desc() const { return "square loss"; }
	};
}

#endif // NANOCV_LOSS_SQUARE_H
