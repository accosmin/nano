#pragma once

#include "sample.h"

namespace ncv
{
        ///
        /// \brief sampling from a potentially large collection of samples.
        ///
        class NANOCV_PUBLIC sampler_t
	{
        public:

                enum class stype : int
                {
                        batch,          ///< use all samples
                        uniform         ///< use a fixed number of uniformly-sampled samples
                };

                enum class atype : int
                {
                        unlabeled,      ///< un-labeled samples
                        annotated       ///< annotated samples
                };

                ///
                /// \brief constructor
                ///
                explicit sampler_t(const samples_t& samples);

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
                /// \brief use the given percentage of samples (other's are the rest of the samples)
                ///
                sampler_t& split(size_t percentage, sampler_t& other);

                ///
                /// \brief reset restrictions (use all samples of the source task)
                ///
                void reset();

                ///
                /// \brief return a set of samples
                ///
                samples_t get() const;

                ///
                /// \brief return the pool of samples
                ///
                const samples_t& all() const { return m_samples; }

                ///
                /// \brief check if any samples available
                ///
                bool empty() const { return all().empty(); }

                ///
                /// \brief return the number of available samples
                ///
                size_t size() const { return all().size(); }

                ///
                /// \brief return if the samples are selected randomly
                ///
                bool is_random() const { return m_stype != stype::batch; }

        private:

                ///
                /// \brief order samples for fast caching
                ///
                sampler_t& order();

        private:

                // attributes
                samples_t               m_osamples;             ///< original sample pool
                samples_t               m_samples;              ///< current pool of samples

                stype                   m_stype;
                size_t                  m_ssize;
        };
}

