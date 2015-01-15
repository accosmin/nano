#include "sample.h"
#include <set>

namespace ncv
{
        strings_t labels(const samples_t& samples)
        {
                return ncv::labels(samples.begin(), samples.end());
        }

        strings_t labels(samples_const_it_t begin, samples_const_it_t end)
        {
                std::set<string_t> label_ids;
                for (auto it = begin; it != end; ++ it)
                {
                        const sample_t& sample = *it;
//                        if (sample.annotated() &&  !sample.m_label.empty())
                        {
                                label_ids.insert(sample.m_label);
                        }
                }

                return strings_t(label_ids.begin(), label_ids.end());
        }
}
