#include "ncv_loss.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        vector_t class_target(index_t ilabel, size_t n_labels)
        {
                vector_t target(n_labels);
                target.setConstant(neg_target());
                target[ilabel] = pos_target();
                return target;
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t l1_error(const vector_t& targets, const vector_t& scores)
        {
                return (targets - scores).array().abs().sum();
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t eclass_error(const vector_t& targets, const vector_t& scores)
        {
                return (targets.array() * scores.array() <= std::numeric_limits<scalar_t>::epsilon()).count();
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t mclass_error(const vector_t& targets, const vector_t& scores)
        {
                std::ptrdiff_t idx = 0;
                scores.maxCoeff(&idx);

                return targets(idx) > 0.5 ? 0.0 : 1.0;
        }

        //-------------------------------------------------------------------------------------------------
}

