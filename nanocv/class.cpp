#include "class.h"

namespace ncv
{
        vector_t class_target(size_t ilabel, size_t n_labels)
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
	
