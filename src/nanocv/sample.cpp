#include "sample.h"
#include <numeric>
#include <map>
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
                        label_ids.insert(sample.m_label);
                }

                return strings_t(label_ids.begin(), label_ids.end());
        }

        bool label_normalize(samples_t& samples)
        {
                return ncv::label_normalize(samples.begin(), samples.end());
        }

        bool label_normalize(samples_it_t begin, samples_it_t end)
        {
                std::map<string_t, size_t> label_counts;
                for (auto it = begin; it != end; ++ it)
                {
                        const sample_t& sample = *it;
                        label_counts[sample.m_label] ++;
                }

                const scalar_t scalew = (0.0 + std::distance(begin, end)) / (0.0 + label_counts.size());

                for (auto it = begin; it != end; ++ it)
                {
                        sample_t& sample = *it;
                        sample.m_weight = scalew / label_counts[sample.m_label];
                }

                return true;
        }

        bool normalize(samples_t& samples)
        {
                return ncv::normalize(samples.begin(), samples.end());
        }

        bool normalize(samples_it_t begin, samples_it_t end)
        {
                const scalar_t scalew = std::distance(begin, end) / ncv::accumulate(begin, end);

                for (auto it = begin; it != end; ++ it)
                {
                        sample_t& sample = *it;
                        sample.m_weight *= scalew;
                }

                return true;
        }

        scalar_t accumulate(const samples_t& samples)
        {
                return ncv::accumulate(samples.begin(), samples.end());
        }

        scalar_t accumulate(samples_const_it_t begin, samples_const_it_t end)
        {
                return  std::accumulate(begin, end, 0.0,
                        [] (double sum, const sample_t& sample) { return sum + sample.weight(); });
        }
}
