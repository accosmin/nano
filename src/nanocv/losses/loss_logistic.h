#ifndef NANOCV_LOSS_LOGISTIC_H
#define NANOCV_LOSS_LOGISTIC_H

#include "loss.h"

namespace ncv
{
        ///
        /// \brief summed (multi-class) logistic loss
	///
        class sum_logistic_loss_t : public loss_t
        {
        public:

                // constructor
                sum_logistic_loss_t();

                // create an object clone
                virtual rloss_t clone(const string_t&) const { return rloss_t(new sum_logistic_loss_t); }

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const
                {
                        return multi_class_error(targets, scores);
                }

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const;
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const;
        };

        ///
        /// \brief soft-max (multi-class) logistic loss
        ///
        class max_logistic_loss_t : public loss_t
        {
        public:

                // constructor
                max_logistic_loss_t();

                // create an object clone
                virtual rloss_t clone(const string_t&) const { return rloss_t(new max_logistic_loss_t); }

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

#endif // NANOCV_LOSS_LOGISTIC_H
