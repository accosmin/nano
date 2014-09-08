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
        /// parameters:
        ///     alpha=1[1,10]           - approximates the sum/avg (if close to 1) or the max (if large, e.g. 10)
        ///
        class logistic_loss_t : public loss_t
        {
        public:

                NANOCV_MAKE_CLONABLE(logistic_loss_t)

                // constructor
                logistic_loss_t(const string_t& parameters = string_t())
                        :       loss_t(string_t(), "multi-class logistic loss, parameters: alpha=1.0[1.0,10.0]"),
                                m_alpha(math::clamp(text::from_params<scalar_t>(parameters, "alpha", 1.0), 1.0, 10.0))
                {
                }

                // compute the error value
                virtual scalar_t error(const vector_t& targets, const vector_t& scores) const
                {
                        return mclass_error(targets, scores);
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
                        
                        const vector_t edges = (- m_alpha * targets.array() * scores.array()).exp();

                        return std::log(1.0 + edges.sum() / scores.size()) / m_alpha;
                }

                vector_t _vgrad(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());
                        
                        const vector_t edges = (- m_alpha * targets.array() * scores.array()).exp();
                        
                        return - targets.array() * edges.array() / (scores.size() + edges.sum());
                }

        private:

                // attributes
                scalar_t        m_alpha;        ///< scaling factor (controls the max/avg-like effect)
        };
}

#endif // NANOCV_LOSS_LOGISTIC_HPP
