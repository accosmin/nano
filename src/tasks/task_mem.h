#pragma once

#include "task.h"
#include "core/random.h"

namespace nano
{
        ///
        /// \brief in-memory task consisting of samples organized by fold.
        ///
        /// tchunk is a data piece (e.g. image, tensor)
        ///
        /// tsample is a sample associated to a chunk (e.g. can map to the whole or a part of the chunk):
        ///     ::index()                       - index of the associated chunk
        ///     ::input(const tchunk&)          - input 3D tensor
        ///     ::ihash(size_t chunk_hash)      - hash of the input tensor given the hash of the associated chunk
        ///     ::output()                      - output/target 3D tensor
        ///     ::ohash()                       - hash of the output tensor
        ///     ::label()                       - associated label (if any)
        ///
        template <typename tchunk, typename tsample>
        class mem_task_t : public task_t
        {
        public:
                mem_task_t(const tensor3d_dim_t& idims, const tensor3d_dim_t& odims, const size_t fsize);

                bool load() final;

                tensor3d_dim_t idims() const final { return m_idims; }
                tensor3d_dim_t odims() const final { return m_odims; }

                size_t size() const final;
                size_t size(const fold_t&) const final;
                size_t fsize() const final { return m_fsize; }

                size_t ihash(const fold_t&, const size_t index) const final;
                size_t ohash(const fold_t&, const size_t index) const final;
                string_t label(const fold_t&, const size_t index) const final;

                tensor3d_t input(const fold_t&, const size_t index) const final;
                tensor3d_t target(const fold_t&, const size_t index) const final;

                void shuffle(const fold_t&) const final;

        protected:

                void reconfig(const tensor3d_dim_t& idims, const tensor3d_dim_t& odims, const size_t fsize)
                {
                        m_idims = idims;
                        m_odims = odims;
                        m_fsize = fsize;

                        m_chunks.clear();
                        m_samples.clear();
                }

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
                        assert(fold.m_index < fsize());
                        m_samples[fold].emplace_back(ts...);
                }

                void add_samples(const protocol p, const std::vector<tensor_size_t>& ilabels, const strings_t& tlabels)
                {
                        assert(n_chunks() >= ilabels.size());

                        const auto chunk_begin = n_chunks() - ilabels.size();

                        for (size_t f = 0; f < fsize(); ++ f)
                        {
                                const auto protocols = (p == protocol::train) ?
                                        split2(ilabels.size(), protocol::train, 80, protocol::valid) :
                                        std::vector<protocol>(ilabels.size(), protocol::test);

                                for (size_t i = 0; i < ilabels.size(); ++ i)
                                {
                                        const auto ilabel = ilabels[i];
                                        const auto ichunk = chunk_begin + i;
                                        const auto fold = fold_t{f, protocols[i]};
                                        const auto itlabel = static_cast<size_t>(ilabel);
                                        assert(ilabel >= 0 && itlabel < tlabels.size());

                                        add_sample(fold, ichunk, class_target(nano::size(odims()), ilabel), tlabels[itlabel]);
                                }
                        }
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
                tensor3d_dim_t                  m_idims;        ///< input size
                tensor3d_dim_t                  m_odims;        ///< output size
                size_t                          m_fsize;        ///< number of folds
                std::vector<tchunk>             m_chunks;       ///<
                std::vector<size_t>             m_hashes;       ///< hash / chunk
                mutable tsamples                m_samples;      ///< stored samples (training, validation, test)
        };

        template <typename tchunk, typename tsample>
        mem_task_t<tchunk, tsample>::mem_task_t(
                const tensor3d_dim_t& idims,
                const tensor3d_dim_t& odims,
                const size_t fsize) :
                m_idims(idims),
                m_odims(odims),
                m_fsize(fsize)
        {
        }

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
        size_t mem_task_t<tchunk, tsample>::size() const
        {
                return  std::accumulate(m_samples.begin(), m_samples.end(), size_t(0),
                        [&] (const size_t count, const auto& samples) { return count + samples.second.size(); });
        }

        template <typename tchunk, typename tsample>
        size_t mem_task_t<tchunk, tsample>::size(const fold_t& fold) const
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
                std::shuffle(it->second.begin(), it->second.end(), make_rng());
        }

        template <typename tchunk, typename tsample>
        tensor3d_t mem_task_t<tchunk, tsample>::input(const fold_t& fold, const size_t index) const
        {
                assert(index < size(fold));
                const auto& sample = get_sample(fold, index);
                const auto& chunk = get_chunk(sample);
                return sample.input(chunk);
        }

        template <typename tchunk, typename tsample>
        tensor3d_t mem_task_t<tchunk, tsample>::target(const fold_t& fold, const size_t index) const
        {
                assert(index < size(fold));
                const auto& sample = get_sample(fold, index);
                return sample.output();
        }

        template <typename tchunk, typename tsample>
        size_t mem_task_t<tchunk, tsample>::ihash(const fold_t& fold, const size_t index) const
        {
                const auto& sample = get_sample(fold, index);
                return sample.ihash(get_hash(sample));
        }

        template <typename tchunk, typename tsample>
        size_t mem_task_t<tchunk, tsample>::ohash(const fold_t& fold, const size_t index) const
        {
                const auto& sample = get_sample(fold, index);
                return sample.ohash();
        }

        template <typename tchunk, typename tsample>
        string_t mem_task_t<tchunk, tsample>::label(const fold_t& fold, const size_t index) const
        {
                const auto& sample = get_sample(fold, index);
                return sample.label();
        }
}
