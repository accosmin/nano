#ifndef NANOCV_LOSS_LOGISTIC_HPP
#define NANOCV_LOSS_LOGISTIC_HPP

#include "loss.h"
#include "common/math.hpp"
#include <cassert>

namespace ncv
{
        ///
        /// \brief multi-class logistic loss
        ///
        class logistic_loss_t : public loss_t
        {
        public:

                NANOCV_MAKE_CLONABLE(logistic_loss_t)

                // constructor
                logistic_loss_t(const string_t& = string_t())
                        :       loss_t(string_t(), "multi-class logistic loss")
                {
                }

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const
                {
                        return mclass_edge_error(targets, scores);
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
                        
                        const vector_t edges = (- targets.array() * scores.array()).exp();
                        
                        return std::log(1.0 + edges.sum());
                }

                vector_t _vgrad(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());
                        
                        const vector_t edges = (- targets.array() * scores.array()).exp();
                        const scalar_t norm = 1.0 / (1.0 + edges.sum());
                        
                        return - targets.array() * edges.array() * norm;
                }
        };
}

#endif // NANOCV_LOSS_LOGISTIC_HPP
