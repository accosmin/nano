#include "loss_classnll.h"
#include "cortex/class.h"
#include "math/numeric.hpp"
#include <cassert>

namespace cortex
{
        classnll_loss_t::classnll_loss_t(const string_t& configuration)
                :       loss_t(configuration)
        {
        }

        scalar_t classnll_loss_t::error(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());
                
                vector_t::Index idx;
                scores.maxCoeff(&idx);
                
                return is_pos_target(targets(idx)) ? 0.0 : 1.0;
        }

        scalar_t classnll_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());
                
                const vector_t escores = scores.array().exp();
                
                return std::log(escores.array().sum()) - 0.5 * (1.0 + targets.array()).matrix().dot(scores);
        }
        
        vector_t classnll_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());
                
                const vector_t escores = scores.array().exp();
                
                return escores.array() / escores.sum() - 0.5 * (1.0 + targets.array());
        }

        indices_t classnll_loss_t::labels(const vector_t& scores) const
        {
                vector_t::Index idx;
                scores.maxCoeff(&idx);
                
                return indices_t(1, size_t(idx));
        }
}

