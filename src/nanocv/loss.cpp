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

        scalar_t mclass_error(const vector_t& targets, const vector_t& scores)
        {
                assert(targets.size() == scores.size());
                
                const scalar_t thres = 1.0 / scores.size();
                
                size_t errors = 0;                
                for (auto i = 0; i < scores.size(); i ++)
                {
			const bool target_pos = is_pos_target(targets(i));
			const bool scores_pos = scores(i) > thres;

			if (target_pos != scores_pos)
			{
				errors ++;
                        }
                }
                
                return errors;
        }

        indices_t classes(const vector_t& scores)
        {
                const scalar_t thres = 1.0 / scores.size();

                indices_t ret;
                for (auto i = 0; i < scores.size(); i ++)
                {
                        if (scores(i) > thres)
                        {
                                ret.push_back(i);
                        }
                }

                return ret;
        }
}
	
