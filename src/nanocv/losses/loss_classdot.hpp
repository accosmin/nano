#ifndef NANOCV_LOSS_CLASS_DOT_HPP
#define NANOCV_LOSS_CLASS_DOT_HPP

#include "loss.h"
#include <cassert>

namespace ncv
{
        ///
        /// \brief multi-class positive vs negative difference loss.
        ///
        /// NB: assumes {-1, +1} targets.
        ///
        template
        <
                bool tnormalized_inputs         ///< [0, 1] normalized input scores?!
        >
        class classdot_loss_t : public loss_t
        {
        public:

                // constructor
                classdot_loss_t()
                        :       loss_t(string_t(), "class dot-product loss")
                {
                }

                // create an object clone
                virtual rloss_t clone(const string_t&) const
                {
                        return rloss_t(new classdot_loss_t);
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
                                        // inputs are [0, 1] normalized
                                        return -targets.dot(scores);
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
                                        // inputs are [0, 1] normalized
                                        return -targets;
                                }

                        case false:
                                {
                                        // inputs are not normalized (so normalize the loss value)
                                        const vector_t escores = scores.array().exp();
                                        const scalar_t est = targets.dot(escores);
                                        const scalar_t ess = escores.sum();

                                        return escores.array() * (est / (ess * ess) - targets.array() / ess);
                                }
                        }
                }
        };
}

#endif // NANOCV_LOSS_CLASS_DOT_HPP
