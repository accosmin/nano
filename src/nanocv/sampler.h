#ifndef NANOCV_SAMPLER_H
#define NANOCV_SAMPLER_H

#include "image.h"

namespace ncv
{
        class task_t;

        ///
        /// \brief generic computer vision task consisting of a set of (annotated) images
        /// and a protocol (training + testing).
        /// samples for training & testing models can be drawn from these image.
        ///
        class sampler_t
	{
        public:

                enum class stype : int
                {
                        batch,          ///< use all samples
                        usampler        ///< use a fixed number of uniform samples
                };

                enum class atype : int
                {
                        unlabeled,      ///< un-labeled samples
                        annotated       ///< annotated samples
                };

                ///
                /// \brief constructor
                ///
                sampler_t(const task_t& task);

                ///
                /// \brief restrict by fold
                ///
                sampler_t& setup(fold_t fold);

                ///
                /// \brief restrict by protocol
                ///
                sampler_t& setup(protocol p);

                ///
                /// \brief restrict by sampling type
                ///
                sampler_t& setup(stype s, size_t size = 0);

                ///
                /// \brief restrict by annotation type
                ///
                sampler_t& setup(atype a);

                ///
                /// \brief restrict to the given label
                ///
                sampler_t& setup(const string_t& label);

                ///
                /// \brief restrict to the given samples
                ///
                sampler_t& setup(const samples_t& samples);

                ///
                /// \brief use the given percentage of samples (others are the rest of the samples)
                ///
                sampler_t& split(scalar_t percentage, samples_t& others);

                ///
                /// \brief reset restrictions (use all samples of the source task)
                ///
                void reset();

                ///
                /// \brief return a set of samples
                ///
                samples_t get() const;

        private:

                ///
                /// \brief order samples for fast caching
                ///
                static void order(samples_t& samples);

        private:

                // attributes
                const task_t&           m_task;                 ///< source task
                samples_t               m_samples;              ///< current pool of samples
        };
}

#endif // NANOCV_SAMPLER_H
