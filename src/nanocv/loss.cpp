#include "loss.h"
#include <cassert>

namespace ncv
{
        vector_t class_target(size_t ilabel, size_t n_labels)
        {
                assert(ilabel < n_labels);
                
                vector_t target(n_labels);
                target.setConstant(neg_target());
                if (ilabel < n_labels)
                {
                        target[ilabel] = pos_target();
                }
                return target;
        }

        scalar_t l1_error(const vector_t& targets, const vector_t& scores)
        {
                assert(targets.size() == scores.size());
                
                return (targets - scores).array().abs().sum();
        }

        scalar_t mclass_edge_error(const vector_t& targets, const vector_t& scores)
        {
                assert(targets.size() == scores.size());
                
                size_t errors = 0;                
                for (auto i = 0; i < scores.size(); i ++)
                {
                        const scalar_t edge = targets(i) * scores(i);
                        if (edge <= 0.0)
                        {
                                errors ++;
                        }
                }
                
                return (errors > 0) ? 1.0 : 0.0;
        }

        scalar_t mclass_argmax_error(const vector_t& targets, const vector_t& scores)
        {
                assert(targets.size() == scores.size());

                vector_t::Index idx;
                scores.maxCoeff(&idx);

                return is_pos_target(targets(idx)) ? 0.0 : 1.0;
        }

        indices_t classes(const vector_t& scores)
        {
                indices_t ret;
                for (auto i = 0; i < scores.size(); i ++)
                {
                        if (scores(i) > 0.0)
                        {
                                ret.push_back(i);
                        }
                }

                return ret;
        }
}
	
