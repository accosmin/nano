#ifndef NANOCV_WEIGHTED_SAMPLING_H
#define NANOCV_WEIGHTED_SAMPLING_H

#include <vector>
#include "math.hpp"
#include "random.hpp"

namespace ncv
{
        namespace detail
        {
                ///
                /// \brief create a weighted-based subset of indices
                /// NB: assumming that the weights are normalized (sum = #samples)
                ///
                template
                <
                        typename tsample
                >
                inline std::vector<std::size_t> weighted_indices(const std::vector<tsample>& samples, std::size_t size)
                {
                        const std::size_t capacity = samples.size();
                        const double scalew = (0.0 + size) / (0.0 + capacity);

                        random_t<double> rng(0.0, 1.0);

                        std::vector<std::size_t> indices;
                        for (std::size_t i = 0; i < samples.size(); i ++)
                        {
                                const tsample& sample = samples[i];

                                double prob = sample.weight() * scalew;
                                std::size_t cnt = static_cast<std::size_t>(prob);

                                if (cnt > 0)
                                {
                                        indices.insert(indices.end(), cnt, i);
                                        prob -= cnt;
                                }

                                if (rng() < prob)
                                {
                                        indices.push_back(i);
                                }
                        }

                        std::sort(indices.begin(), indices.end());

                        return indices;
                }
        }

        ///
        /// \brief create a random subset of given size by uniformly sampling the set
        ///
        template
        <
                typename tsample
        >
        std::vector<tsample> weighted_sample(const std::vector<tsample>& samples, std::size_t size)
        {
                std::vector<tsample> usamples;

                if (!samples.empty())
                {
                        const std::vector<std::size_t> indices = detail::weighted_indices(samples, size);

                        for (std::size_t i = 0; i < indices.size(); i ++)
                        {
                                const tsample& sample = samples[indices[i]];
                                usamples.emplace_back(sample);
                        }
                }

                return usamples;
        }

        ///
        /// \brief run a functor for a random subset of given size by uniformly sampling the set
        ///
        template
        <
                typename tsample,
                typename toperator
        >
        void weighted_sample(const std::vector<tsample>& samples, std::size_t size, const toperator& op)
        {
                if (!samples.empty())
                {
                        const std::vector<std::size_t> indices = detail::weighted_indices(samples, size);

                        for (std::size_t i = 0; i < indices.size(); i ++)
                        {
                                const tsample& sample = samples[indices[i]];
                                op(sample);
                        }
                }
        }
}

#endif // NANOCV_WEIGHTED_SAMPLING_H

