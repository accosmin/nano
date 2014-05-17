#ifndef NANOCV_LOSS_CLASS_NLL_HPP
#define NANOCV_LOSS_CLASS_NLL_HPP

#include "loss.h"
#include <cassert>

namespace ncv
{
        ///
        /// \brief multi-class negative log-likelihood loss
        ///
        class classnll_loss_t : public loss_t
        {
        public:

                // constructor
                classnll_loss_t()
                        :       loss_t(string_t(), "class negative-log-likelihood loss")
                {
                }

                // create an object clone
                virtual rloss_t clone(const string_t&) const
                {
                        return rloss_t(new classnll_loss_t);
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

                        return  std::log(scores.array().exp().sum()) -
                                0.5 * (1.0  + targets.array()).matrix().dot(scores);
                }

                vector_t _vgrad(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());

                        return  scores.array().exp().matrix() / scores.array().exp().sum() -
                                0.5 * (1.0  + targets.array()).matrix();
                }
        };
}

#endif // NANOCV_LOSS_CLASS_NLL_HPP
