#pragma once

#include "loss.h"
#include "common/math.hpp"
#include <cassert>

namespace ncv
{
        ///
        /// \brief multi-class logistic loss (multi-label)
        ///
        class logistic_loss_t : public loss_t
        {
        public:

                NANOCV_MAKE_CLONABLE(logistic_loss_t)

                // constructor
                logistic_loss_t(const string_t& parameters = string_t())
                        :       loss_t(string_t(), "multi-class logistic loss")
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

                        size_t errors = 0;
                        for (auto i = 0; i < scores.size(); i ++)
                        {
                                const scalar_t edge = targets(i) * scores(i);
                                if (edge <= 0.0)
                                {
                                        errors ++;
                                }
                        }

                        return errors;
                }

                scalar_t _value(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());

                        const vector_t edges = (- targets.array() * scores.array()).exp();

                        return std::log(1.0 + edges.sum());
                }

                vector_t _vgrad(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());
                        
                        const vector_t edges = (- targets.array() * scores.array()).exp();
                        
                        return - targets.array() * edges.array() / (1.0 + edges.sum());
                }

                indices_t _labels(const vector_t& scores) const
                {
                        indices_t ret;
                        for (auto i = 0; i < scores.size(); i ++)
                        {
                                if (scores(i) > 0.0)
                                {
                                        ret.push_back(i);
                                }
                        }

                        return ret;
                }
        };
}
