#ifndef NANOCV_LOSS_SPARSE_H
#define NANOCV_LOSS_SPARSE_H

#include "loss.h"

namespace ncv
{
        ///
        /// \brief output sparsity-enforcing loss (few outputs are close to -1/+1, the rest to 0)
	///
	/// parameters:
        ///     w=1[0,1000]		- weight closeness to -1/+1 (low) vs closeness to 0 (high)
	///
        class sparse_loss_t : public loss_t
        {
        public:

                // constructor
                sparse_loss_t(const string_t& params = string_t());

                // create an object clone
                virtual rloss_t clone(const string_t& params) const { return rloss_t(new sparse_loss_t(params)); }

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const
                {
                        return eclass_error(targets, scores);
                }

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const;

	private:
		
		// attributes
		scalar_t		m_weight;	///< "w" parameter that weights the two components
							///	(1) - closeness to -1/+1
							/// 	(2) - sparsity (~ #outputs close to 0)
        };
}

#endif // NANOCV_LOSS_SPARSE_H
