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

        vector_t class_target(const tensor_index_t ilabel, const tensor_size_t n_labels)
        {
                vector_t target(n_labels);
                target.setConstant(neg_target());
                if (ilabel < n_labels)
                {
                        target[ilabel] = pos_target();
                }
                return target;
        }

        indices_t class_labels(const vector_t& scores)
        {
                indices_t ret;
                for (auto i = 0; i < scores.size(); ++ i)
                {
                        if (scores(i) > 0.0)
                        {
                                ret.push_back(static_cast<size_t>(i));
                        }
                }

                return ret;
        }
}

