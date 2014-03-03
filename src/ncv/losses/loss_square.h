#ifndef NANOCV_LOSS_SQUARE_H
#define NANOCV_LOSS_SQUARE_H

#include "loss.h"

namespace ncv
{
        ///
        /// \brief square loss
        ///
        class square_loss_t : public loss_t
	{
	public:

                // constructor
                square_loss_t();

                // create an object clone
                virtual rloss_t clone(const string_t&) const { return rloss_t(new square_loss_t); }

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const
                {
                        return l1_error(targets, scores);
                }

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const;
	};
}

#endif // NANOCV_LOSS_SQUARE_H
