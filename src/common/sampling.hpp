#pragma once

#include "detail/sampling_impl.hpp"

namespace ncv
{
        ///
        /// \brief create a random subset of given size by uniformly sampling the set
        ///
        template
        <
                typename tsample
        >
        std::vector<tsample> uniform_sample(const std::vector<tsample>& samples, std::size_t size)
        {
                std::vector<tsample> usamples;

                if (!samples.empty())
                {
                        const std::vector<std::size_t> indices = detail::uniform_indices(samples.size(), size);

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
        void uniform_sample(const std::vector<tsample>& samples, std::size_t size, const toperator& op)
        {
                if (!samples.empty())
                {
                        const std::vector<std::size_t> indices = detail::uniform_indices(samples.size(), size);

                        for (std::size_t i = 0; i < indices.size(); i ++)
                        {
                                const tsample& sample = samples[indices[i]];
                                op(sample);
                        }
                }
        }

        ///
        /// \brief split the given samples in two subsets of (percentage1)% and (100 - percentage1)% proportion
        ///
        template
        <
                typename tsample
        >
        void uniform_split(const std::vector<tsample>& samples, std::size_t percentage1,
                std::vector<tsample>& usamples1, std::vector<tsample>& usamples2)
        {
                random_t<std::size_t> rng(0, 99);

                usamples1.clear();
                usamples2.clear();

                for (std::size_t i = 0; i < samples.size(); i ++)
                {
                        (rng() < percentage1 ? usamples1 : usamples2).emplace_back(samples[i]);
                }
        }
}
