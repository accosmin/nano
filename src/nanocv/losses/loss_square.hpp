#ifndef NANOCV_LOSS_SQUARE_HPP
#define NANOCV_LOSS_SQUARE_HPP

#include "loss.h"
#include <cassert>

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
                square_loss_t(const string_t& = string_t())
                        :       loss_t(string_t(), "square loss")
                {
                }

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const
                {
                        return _error(targets, scores);
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

                // predict label indices
                virtual indices_t labels(const vector_t& scores) const
                {
                        return _labels(scores);
                }

        private:

                scalar_t _error(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());

                        return (targets - scores).array().abs().sum();
                }

                scalar_t _value(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());

                        return 0.5 * (scores - targets).array().square().sum();
                }

                vector_t _vgrad(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());

                        return scores - targets;
                }

                indices_t _labels(const vector_t& scores) const
                {
                        return indices_t();
                }
	};
}

#endif // NANOCV_LOSS_SQUARE_HPP
