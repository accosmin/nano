#pragma once

#include "loss.h"

namespace ncv
{
        ///
        /// \brief square loss
        ///
        class square_loss_t : public loss_t
	{
	public:

                NANOCV_MAKE_CLONABLE(square_loss_t)

                // constructor
                square_loss_t(const string_t& = string_t());

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const;

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const;

                // predict label indices
                virtual indices_t labels(const vector_t& scores) const;;
	};
}

