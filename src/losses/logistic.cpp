#include "logistic.h"
#include "class.h"
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

                return std::log1p(std::exp((-targets.array() * scores.array()).sum()));
        }

        vector_t logistic_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return  -targets.array() * std::exp((-targets.array() * scores.array()).sum()) /
                        (1 + std::exp((-targets.array() * scores.array()).sum()));
        }

        indices_t logistic_loss_t::labels(const vector_t& scores) const
        {
                return class_labels(scores);
        }
}
