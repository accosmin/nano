#ifndef NANOCV_LOSS_CLASSNLL_HPP
#define NANOCV_LOSS_CLASSNLL_HPP

#include "loss.h"
#include "common/math.hpp"
#include <cassert>

namespace ncv
{
        ///
        /// \brief multi-class negative log-likelihood loss (single-label)
        ///
        class classnll_loss_t : public loss_t
        {
        public:

                NANOCV_MAKE_CLONABLE(classnll_loss_t)

                // constructor
                classnll_loss_t(const string_t& parameters = string_t())
                        :       loss_t(string_t(), "multi-class negative log-likelihood loss")
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

                        vector_t::Index idx;
                        scores.maxCoeff(&idx);

                        return is_pos_target(targets(idx)) ? 0.0 : 1.0;
                }

                scalar_t _value(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());

                        const vector_t escores = scores.array().exp();

                        return std::log(escores.array().sum()) - 0.5 * (1.0 + targets.array()).matrix().dot(scores);
                }

                vector_t _vgrad(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());

                        const vector_t escores = scores.array().exp();

                        return escores.array() / escores.sum() - 0.5 * (1.0 + targets.array());
                }

                indices_t _labels(const vector_t& scores) const
                {
                        vector_t::Index idx;
                        scores.maxCoeff(&idx);

                        return indices_t(1, size_t(idx));
                }
        };
}

#endif // NANOCV_LOSS_CLASSNLL_HPP
