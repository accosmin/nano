#include "class.h"

namespace cortex
{
        scalar_t pos_target()
        {
                return +1.0;
        }

        scalar_t neg_target()
        {
                return -1.0;
        }

        bool is_pos_target(const scalar_t target)
        {
                return target > 0.5;
        }

        vector_t class_target(const size_t ilabel, const size_t n_labels)
        {
                vector_t target(n_labels);
                target.setConstant(neg_target());
                if (ilabel < n_labels)
                {
                        target[ilabel] = pos_target();
                }
                return target;
        }
}
	
