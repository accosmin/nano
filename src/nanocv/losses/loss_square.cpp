#include "loss_square.h"
#include <cassert>

namespace ncv
{
        square_loss_t::square_loss_t(const string_t& configuration)
                :       loss_t(configuration)
        {
        }

        scalar_t square_loss_t::error(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());
                
                return (targets - scores).array().abs().sum();
        }

        scalar_t square_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());
                
                return 0.5 * (scores - targets).array().square().sum();
        }
        
        vector_t square_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());
                
                return scores - targets;
        }

        indices_t square_loss_t::labels(const vector_t& scores) const
        {
                return indices_t();
	}
}

