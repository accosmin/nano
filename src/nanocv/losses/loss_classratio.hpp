#ifndef NANOCV_LOSS_CLASS_RATIO_HPP
#define NANOCV_LOSS_CLASS_RATIO_HPP

#include "loss.h"
#include "common/math.hpp"
#include <cassert>

namespace ncv
{
        ///
        /// \brief multi-class loss: minimize the ratio of the negative outputs to the positive outputs.
        ///
        /// NB: assumes {+0, +1} targets.
        ///
        template
        <
                bool tnormalized_inputs         ///< (0, 1) normalized input scores?
        >
        class classratio_loss_t : public loss_t
        {
        public:

                NANOCV_MAKE_CLONABLE(classratio_loss_t)

                // constructor
                classratio_loss_t(const string_t& = string_t())
                        :       loss_t(string_t(), "multi-class ratio loss")
                {
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

                static scalar_t eps() { return 1e-1; }
                static scalar_t elf() { return 1e-1; }

                scalar_t _value(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());

                        switch (tnormalized_inputs)
                        {
                        case false:     // inputs are not normalized (so normalize the loss value)
                                {
                                        const vector_t escores = (elf() * scores.array()).exp();
                                        const vector_t neg = (1.0 - targets.array()) * escores.array();
                                        const vector_t pos = (0.0 + targets.array()) * escores.array();

                                        const scalar_t sumneg = eps() + neg.sum();
                                        const scalar_t sumpos = eps() + pos.sum();

                                        return  sumneg / sumpos;
                                }

                        case true:      // inputs are (0, 1) normalized
                                {
                                        const vector_t neg = (1.0 - targets.array()) * scores.array();
                                        const vector_t pos = (0.0 + targets.array()) * scores.array();

                                        const scalar_t sumneg = eps() + neg.sum();
                                        const scalar_t sumpos = eps() + pos.sum();

                                        return  sumneg / sumpos;
                                }
                        }
                }

                vector_t _vgrad(const vector_t& targets, const vector_t& scores) const
                {
                        assert(targets.size() == scores.size());

                        switch (tnormalized_inputs)
                        {
                        case false:     // inputs are not normalized (so normalize the loss value)
                                {
                                        const vector_t escores = (elf() * scores.array()).exp();
                                        const vector_t neg = (1.0 - targets.array()) * escores.array();
                                        const vector_t pos = (0.0 + targets.array()) * escores.array();

                                        const scalar_t sumneg = eps() + neg.sum();
                                        const scalar_t sumpos = eps() + pos.sum();

                                        return  elf() * escores.array() *
                                                ((1.0 - targets.array()) * sumpos -
                                                 (0.0 + targets.array()) * sumneg) / math::square(sumpos);
                                }                                

                        case true:      // inputs are (0, 1) normalized
                                {
                                        const vector_t neg = (1.0 - targets.array()) * scores.array();
                                        const vector_t pos = (0.0 + targets.array()) * scores.array();

                                        const scalar_t sumneg = eps() + neg.sum();
                                        const scalar_t sumpos = eps() + pos.sum();

                                        return  ((1.0 - targets.array()) * sumpos -
                                                 (0.0 + targets.array()) * sumneg) / math::square(sumpos);
                                }
                        }
                }
        };
}

#endif // NANOCV_LOSS_CLASS_RATIO_HPP
