#ifndef NANOCV_UNIFORM_H
#define NANOCV_UNIFORM_H

#include <vector>
#include "math.hpp"

namespace ncv
{
        ///
        /// create a random subset of given size by uniformly sampling the set
        ///
        template
        <
                typename tsample,
                typename tsize,
                typename tgenerator
        >
        std::vector<tsample> uniform_sample(const std::vector<tsample>& samples, tsize size, tgenerator rng)
        {
                std::vector<tsample> usamples;

                if (!samples.empty())
                {
                        for (tsize i = 0; i < size; i ++)
                        {
                                const tsample& sample = samples[rng() % samples.size()];
                                usamples.emplace_back(sample);
                        }
                }

                return usamples;
        }

        ///
        /// run a functor for a random subset of given size by uniformly sampling the set
        ///
        template
        <
                typename tsample,
                typename tsize,
                typename tgenerator,
                typename toperator
        >
        void uniform_sample(const std::vector<tsample>& samples, tsize size, tgenerator rng, const toperator& op)
        {
                if (!samples.empty())
                {
                        for (tsize i = 0; i < size; i ++)
                        {
                                const tsample& sample = samples[rng() % samples.size()];
                                op(sample);
                        }
                }
        }

        ///
        /// split the given samples in two subsets of (percentage1)% and (100 - percentage1)% proportion
        ///
        template
        <
                typename tsample,
                typename tsize,
                typename tgenerator
        >
        void uniform_split(const std::vector<tsample>& samples, tsize percentage1, tgenerator rng,
                std::vector<tsample>& usamples1, std::vector<tsample>& usamples2)
        {
                usamples1.clear();
                usamples2.clear();

                percentage1 = math::clamp(percentage1, tsize(1), tsize(99));
                for (tsize i = 0; i < samples.size(); i ++)
                {
                        ((rng() % 100) < percentage1 ? usamples1 : usamples2).emplace_back(samples[i]);
                }
        }
}

#endif // NANOCV_UNIFORM_H

