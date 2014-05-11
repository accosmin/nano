#ifndef NANOCV_LOSS_CLASS_SUM_HPP
#define NANOCV_LOSS_CLASS_SUM_HPP

#include "loss.h"
#include <cassert>

namespace ncv
{
        ///
        /// \brief multi-class positive vs negative difference loss
        ///
        template
        <
                bool tnormalized_inputs         ///< normalized input scores?!
        >
        class classsum_loss_t : public loss_t
        {
        public:

                // constructor
                classsum_loss_t()
                        :       loss_t(string_t(), "class positive vs negative difference loss")
                {
                }

                // create an object clone
                virtual rloss_t clone(const string_t&) const
                {
                        return rloss_t(new classsum_loss_t);
                }

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const
                {
                        return multi_class_error(targets, scores);
                }

                // compute the loss value & derivatives
                virtual scalar_t value(const vector_t& targets, const vector_t& scores) const
                {
                        return _value(targets, scores);
                }
                virtual vector_t vgrad(const vector_t& targets, const vector_t& scores) const
                {
                        return _vgrad(targets, scores);
                }

        private:

                scalar_t _value(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());

                        switch (tnormalized_inputs)
                        {
                        case true:
                                {
                                        // inputs are normalized ([0,1] or [-1,1])
                                        return -targets.dot(scores.array().matrix());
                                }

                        case false:
                                {
                                        // inputs are not normalized (so normalize the loss value)
                                        const vector_t escores = scores.array().exp();

                                        return -targets.dot(escores) / escores.sum();
                                }
                        }
                }

                vector_t _vgrad(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());

                        switch (tnormalized_inputs)
                        {
                        case true:
                                {
                                        // inputs are normalized ([0,1] or [-1,1])
                                        return -targets.array();
                                }

                        case false:
                                {
                                        // inputs are not normalized (so normalize the loss value)
                                        const vector_t escores = scores.array().exp();
                                        const scalar_t est = targets.dot(escores);
                                        const scalar_t ess = escores.sum();

                                        return  escores.array() * (est / (ess * ess) - targets.array() / ess);
                                }
                        }
                }
        };
}

#endif // NANOCV_LOSS_CLASS_SUM_HPP
