#include "loss_classnll.h"
#include "cortex/class.h"
#include "math/numeric.hpp"
#include <cassert>

namespace nano
{
        classnll_loss_t::classnll_loss_t(const string_t& configuration) :
                loss_t(configuration)
        {
        }

        scalar_t classnll_loss_t::error(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                vector_t::Index idx;
                scores.maxCoeff(&idx);

                return is_pos_target(targets(idx)) ? 0 : 1;
        }

        scalar_t classnll_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return  std::log(scores.array().exp().sum()) -
                        scalar_t(0.5) * ((1 + targets.array()) * scores.array()).sum();
        }

        vector_t classnll_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return  scores.array().exp() / (scores.array().exp().sum()) -
                        scalar_t(0.5) * (1 + targets.array());
        }

        indices_t classnll_loss_t::labels(const vector_t& scores) const
        {
                vector_t::Index idx;
                scores.maxCoeff(&idx);

                return { static_cast<size_t>(idx) };
        }
}

