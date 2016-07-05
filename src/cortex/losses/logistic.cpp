#include "logistic.h"
#include "cortex/class.h"
#include "math/softmax.hpp"
#include <cassert>

namespace nano
{
        logistic_loss_t::logistic_loss_t(const string_t& configuration) :
                loss_t(configuration)
        {
        }

        scalar_t logistic_loss_t::error(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                const auto edges = targets.array() * scores.array();

                return static_cast<scalar_t>((edges < std::numeric_limits<scalar_t>::epsilon()).count());
        }

        scalar_t logistic_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return softmax_value((1 + (-targets.array() * scores.array()).exp()).log());
        }

        vector_t logistic_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return  -targets.array() * (-targets.array() * scores.array()).exp() /
                        (1 + (-targets.array() * scores.array()).exp()) *
                        softmax_vgrad((1 + (-targets.array() * scores.array()).exp()).log());
        }

        indices_t logistic_loss_t::labels(const vector_t& scores) const
        {
                return class_labels(scores);
        }
}
