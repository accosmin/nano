#include "loss_classdif.h"
#include <cassert>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        classdif_loss_t::classdif_loss_t()
                :       loss_t(string_t(), "class positive vs negative difference loss")
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t classdif_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return  (2.0 * targets.array() - 1.0).matrix().dot(scores.array().exp().matrix());
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        
        vector_t classdif_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                return  (2.0 * targets.array() - 1.0) * scores.array().exp();
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
