#pragma once

#include "sample.h"

namespace cortex
{
        ///
        /// \brief sampling from a potentially large collection of samples.
        ///
        class NANOCV_PUBLIC sampler_t
	{
        public:

                ///
                /// \brief constructor
                ///
                explicit sampler_t(const samples_t& samples);

                ///
                /// \brief restrict by fold
                ///
                sampler_t& push(const fold_t);

                ///
                /// \brief restrict by protocol
                ///
                sampler_t& push(const protocol);

                ///
                /// \brief restrict by annotation type
                ///
                sampler_t& push(const annotation);

                ///
                /// \brief restrict to the given label
                ///
                sampler_t& push(const string_t& label);

                ///
                /// \brief setup the number of samples to select at a given time
                ///
                sampler_t& push(const size_t batchsize);

                ///
                /// \brief change the sampling pool
                ///
                sampler_t& push(const samples_t& samples);

                ///
                /// \brief reset the previous filter
                ///
                bool pop();

                ///
                /// \brief split the current selection using the given percentage of samples
                ///
                sampler_t& split(const size_t percentage, sampler_t& other);

                ///
                /// \brief select a set of samples
                ///
                samples_t get() const;

                ///
                /// \brief return the number of available samples
                ///
                size_t size() const;

                ///
                /// \brief check if any available samples
                ///
                bool empty() const;

        private:

                struct state_t
                {
                        explicit state_t(const samples_t& samples = samples_t(),
                                         const size_t batchsize = 0);

                        samples_t       m_samples;              ///< pool of samples to choose from
                        size_t          m_batchsize;            ///< number of samples to choose at a time
                };

                ///
                /// \brief current sampling pool
                ///
                state_t& current();

                ///
                /// \brief current sampling pool
                ///
                const state_t& current() const;

        private:

                // attributes
                std::vector<state_t>    m_states;               ///< filtering/selection history
        };
}

