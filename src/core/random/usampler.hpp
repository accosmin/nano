#ifndef NANOCV_USAMPLER_H
#define NANOCV_USAMPLER_H

#include "random.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // uniform sampling:
        //      the selection probability of a sample is constant.
        /////////////////////////////////////////////////////////////////////////////////////////

        // uniform sampling: create a random subset of given size
        template
        <
                typename tsample,
                typename tsize
        >
        std::vector<tsample> uniform_sample(const std::vector<tsample>& samples, tsize size)
        {
                std::vector<tsample> usamples;

                if (!samples.empty())
                {
                        random_t<tsize> die(tsize(0), static_cast<tsize>(samples.size()) - 1);
                        for (tsize i = 0; i < size; i ++)
                        {
                                const tsample& sample = samples[die()];
                                usamples.emplace_back(sample);
                        }
                }

                return usamples;
        }

        // uniform sampling: run a functor for a random subset of given size
        template
        <
                typename tsample,
                typename tsize,
                typename toperator
        >
        void uniform_sample(const std::vector<tsample>& samples, tsize size, const toperator& op)
        {
                if (!samples.empty())
                {
                        random_t<tsize> die(tsize(0), static_cast<tsize>(samples.size()) - 1);
                        for (tsize i = 0; i < size; i ++)
                        {
                                const tsample& sample = samples[die()];
                                op(sample);
                        }
                }
        }
}

#endif // NANOCV_USAMPLER_H

