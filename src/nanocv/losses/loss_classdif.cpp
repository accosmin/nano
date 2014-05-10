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

//                const vector_t escores = scores.array().exp();

//                return  -targets.dot(escores) / escores.sum();

                return - targets.dot(scores.array().matrix());
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        
        vector_t classdif_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

//                const vector_t escores = scores.array().exp();
//                const scalar_t est = targets.dot(escores);
//                const scalar_t ess = escores.sum();

//                return  escores.array() * (est / (ess * ess) - targets.array() / ess);

                return - targets.array();
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
