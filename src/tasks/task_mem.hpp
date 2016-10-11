#pragma once

#include "task.h"
#include "math/random.hpp"

namespace nano
{
        ///
        /// \brief in-memory task consisting of samples organized by fold.
        ///
        /// tchunk is a data piece (e.g. image, tensor)
        ///
        /// tsample is a sample associated to a chunk (e.g. can map to the whole or a part of the chunk):
        ///     ::index()                       - index of the associated chunk
        ///     ::input(const tchunk&)          - 3D input tensor
        ///     ::input(const size_t chunk_hash)- hash of the input tensor given the hash of the associated chunk
        ///     ::target()                      - target vector
        ///     ::label()                       - associated label (if any)
        ///
        template
        <
                typename tchunk,
                typename tsample
        >
        class mem_task_t : public task_t
        {
        public:

                ///
                /// \brief constructor
                ///
                mem_task_t(
                        const string_t& name,
                        const tensor_size_t idims, const tensor_size_t irows, const tensor_size_t icols,
                        const tensor_size_t osize,
                        const size_t fsize,
                        const string_t& configuration = string_t()) :
                        task_t(configuration),
                        m_name(name),
                        m_idims(idims), m_irows(irows), m_icols(icols), m_osize(osize),
                        m_fsize(fsize), m_frand(1, 10)
                {
                }

                ///
                /// \brief destructor
                ///
                virtual ~mem_task_t() {}

                ///
                /// \brief short name of this task
                ///
                virtual string_t name() const final { return m_name; }

                ///
                /// \brief populate the task with samples
                ///
                virtual bool load() final;

                ///
                /// \brief input size
                ///
                virtual tensor_size_t idims() const final { return m_idims; }
                virtual tensor_size_t irows() const final { return m_irows; }
                virtual tensor_size_t icols() const final { return m_icols; }

                ///
                /// \brief output size
                ///
                virtual tensor_size_t osize() const final { return m_osize; }

                ///
                /// \brief number of folds (not considering the protocol!)
                ///
                virtual size_t n_folds() const final { return m_fsize; }

                ///
                /// \brief total number of samples
                ///
                virtual size_t n_samples() const final;

                ///
                /// \brief number of samples for the given fold
                ///
                virtual size_t n_samples(const fold_t&) const final;

                ///
                /// \brief randomly shuffle the samples associated for the given fold
                ///
                virtual void shuffle(const fold_t&) const final;

                ///
                /// \brief retrieve the 3D input tensor for a given sample
                ///
                virtual tensor3d_t input(const fold_t&, const size_t index) const final;

                ///
                /// \brief retrieve the target for a given sample
                ///
                virtual vector_t target(const fold_t&, const size_t index) const final;

                ///
                /// \brief retrieve the associated label (if any) for a given sample
                ///
                virtual string_t label(const fold_t&, const size_t index) const final;

                ///
                /// \brief retrieve the hash for a given sample
                ///
                virtual size_t hash(const fold_t&, const size_t index) const final;

        protected:

                void reserve_chunks(const size_t count)
                {
                        m_chunks.reserve(count);
                        m_hashes.reserve(count);
                }

                void add_chunk(const tchunk& chunk, const size_t hash)
                {
                        m_chunks.push_back(chunk);
                        m_hashes.push_back(hash);
                }

                template <typename... t>
                void add_sample(const fold_t& fold, t&&... ts)
                {
                        assert(fold.m_index < n_folds());
                        m_samples[fold].emplace_back(ts...);
                }

                fold_t make_fold(const size_t fold) const
                {
                        assert(fold < n_folds());
                        const size_t p = m_frand();
                        // 60% training, 20% validation, 20% testing
                        return {fold, p < 7 ? protocol::train : (p < 9 ? protocol::valid : protocol::test)};
                }

                fold_t make_fold(const size_t fold, const protocol proto) const
                {
                        assert(fold < n_folds());
                        const size_t p = m_frand();
                        // split training into {80% training, 20% validation}, leave the testing as it is
                        return {fold, proto == protocol::train ? (p < 9 ? protocol::train : protocol::valid) : proto};
                }

                virtual bool populate() = 0;

                size_t n_chunks() const { return m_chunks.size(); }
                const tchunk& chunk(const size_t index) const
                {
                        assert(index < n_chunks());
                        return m_chunks[index];
                }

        private:

                using tsamples = std::map<fold_t, std::vector<tsample>>;

                const tsample& get_sample(const fold_t& fold, const size_t sample_index) const
                {
                        const auto it = m_samples.find(fold);
                        assert(it != m_samples.end());
                        assert(sample_index < it->second.size());
                        return it->second[sample_index];
                }

                const tchunk& get_chunk(const tsample& sample) const
                {
                        const auto chunk_index = sample.index();
                        assert(chunk_index < m_chunks.size());
                        return m_chunks[chunk_index];
                }

                const size_t& get_hash(const tsample& sample) const
                {
                        const auto hash_index = sample.index();
                        assert(hash_index < m_hashes.size());
                        return m_hashes[hash_index];
                }

        private:

                // attributes
                string_t                        m_name;
                tensor_size_t                   m_idims;        ///< input size
                tensor_size_t                   m_irows;
                tensor_size_t                   m_icols;
                tensor_size_t                   m_osize;        ///< output size
                size_t                          m_fsize;        ///< number of folds
                mutable random_t<size_t>        m_frand;        ///< rng for training-validation fold assignment
                std::vector<tchunk>             m_chunks;       ///<
                std::vector<size_t>             m_hashes;       ///< hash / chunk
                mutable tsamples                m_samples;      ///< stored samples (training, validation, test)
        };

        template <typename tchunk, typename tsample>
        bool mem_task_t<tchunk, tsample>::load()
        {
                m_chunks.clear();
                m_samples.clear();

                if (!populate())
                {
                        m_chunks.clear();
                        m_samples.clear();
                        return false;
                }
                else
                {
                        // tidy-up memory
                        m_chunks.shrink_to_fit();
                        for (auto& data : m_samples)
                        {
                                data.second.shrink_to_fit();
                        }
                        return true;
                }
        }

        template <typename tchunk, typename tsample>
        size_t mem_task_t<tchunk, tsample>::n_samples() const
        {
                return  std::accumulate(m_samples.begin(), m_samples.end(), size_t(0),
                        [&] (const size_t count, const auto& samples) { return count + samples.second.size(); });
        }

        template <typename tchunk, typename tsample>
        size_t mem_task_t<tchunk, tsample>::n_samples(const fold_t& fold) const
        {
                const auto it = m_samples.find(fold);
                assert(it != m_samples.end());
                return it->second.size();
        }

        template <typename tchunk, typename tsample>
        void mem_task_t<tchunk, tsample>::shuffle(const fold_t& fold) const
        {
                const auto it = m_samples.find(fold);
                assert(it != m_samples.end());

                std::random_device rd;
                std::minstd_rand g(rd());
                std::shuffle(it->second.begin(), it->second.end(), g);
        }

        template <typename tchunk, typename tsample>
        tensor3d_t mem_task_t<tchunk, tsample>::input(const fold_t& fold, const size_t index) const
        {
                const auto& sample = get_sample(fold, index);
                const auto& chunk = get_chunk(sample);
                return sample.input(chunk);
        }

        template <typename tchunk, typename tsample>
        size_t mem_task_t<tchunk, tsample>::hash(const fold_t& fold, const size_t index) const
        {
                const auto& sample = get_sample(fold, index);
                const auto& hash = get_hash(sample);
                return sample.hash(hash);
        }

        template <typename tchunk, typename tsample>
        vector_t mem_task_t<tchunk, tsample>::target(const fold_t& fold, const size_t index) const
        {
                const auto& sample = get_sample(fold, index);
                return sample.target();
        }

        template <typename tchunk, typename tsample>
        string_t mem_task_t<tchunk, tsample>::label(const fold_t& fold, const size_t index) const
        {
                const auto& sample = get_sample(fold, index);
                return sample.label();
        }
}

