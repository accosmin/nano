#include "loss_cauchy.h"
#include <cassert>

namespace ncv
{
        cauchy_loss_t::cauchy_loss_t(const string_t& configuration)
                :       loss_t(configuration)
        {
        }

        scalar_t cauchy_loss_t::error(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());
                
                return (targets - scores).array().abs().sum();
        }

        scalar_t cauchy_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());
                
                return ((targets - scores).array().square() + 1.0).log().sum();
        }
        
        vector_t cauchy_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());
                
                return 2.0 * (scores - targets).array() / (1.0 + (scores - targets).array().square());
        }

        indices_t cauchy_loss_t::labels(const vector_t& scores) const
        {
                NANOCV_UNUSED1(scores);

                return indices_t();
	}
}

