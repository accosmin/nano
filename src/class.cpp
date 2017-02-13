#include "class.h"

namespace nano
{
        scalar_t pos_target()
        {
                return +1;
        }

        scalar_t neg_target()
        {
                return -1;
        }

        bool is_pos_target(const scalar_t target)
        {
                return target > 0;
        }

        vector_t class_target(const tensor_index_t ilabel, const tensor_size_t n_labels)
        {
                vector_t target = vector_t::Constant(n_labels, neg_target());
                if (ilabel < n_labels)
                {
                        target(ilabel) = pos_target();
                }
                return target;
        }

        vector_t class_target(const vector_t& scores)
        {
                vector_t target(scores.size());
                for (auto i = 0; i < scores.size(); ++ i)
                {
                        target(i) = is_pos_target(scores(i)) ? pos_target() : neg_target();
                }
                return target;
        }
}
