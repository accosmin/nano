#pragma once

#include "task.h"
#include "math/random.h"

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
        ///     ::input(const size_t chunk_hash)- hash of the input tensor given the hash of the associated chunk
        ///     ::target()                      - target 3D tensor
        ///     ::label()                       - associated label (if any)
        ///
        template <typename titensor, typename totensor>
        struct mem_task_t : public task_t
        {
                ///
                /// \brief constructor
                ///
                mem_task_t(
                        const tensor3d_dims_t& idims,
                        const tensor3d_dims_t& odims,
                        const size_t fsize,
                        const string_t& params = string_t()) :
                        task_t(params),
                        m_idims(idims), m_odims(odims),
                        m_fsize(fsize), m_frand(1, 10)
                {
                }

                virtual bool load() override final;

                virtual tensor3d_dims_t idims() const override final { return m_idims; }
                virtual tensor3d_dims_t odims() const override final { return m_odims; }

                virtual size_t size() const override final;
                virtual size_t size(const fold_t&) const override final;
                virtual size_t fsize() const override final { return m_fsize; }

                virtual void shuffle(const fold_t&) const override final;

                virtual size_t ihash(const fold_t&, const size_t index) const override final;
                virtual size_t ohash(const fold_t&, const size_t index) const override final;
                virtual sample_t get(const fold_t&, const size_t index) const override final;

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
                        assert(fold.m_index < fsize());
                        m_samples[fold].emplace_back(ts...);
                }

                fold_t make_fold(const size_t fold) const
                {
                        assert(fold < fsize());
                        const size_t p = m_frand();
                        // 60% training, 20% validation, 20% testing
                        return {fold, p < 7 ? protocol::train : (p < 9 ? protocol::valid : protocol::test)};
                }

                fold_t make_fold(const size_t fold, const protocol proto) const
                {
                        assert(fold < fsize());
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

                struct fold_data_t
                {
                        bool valid() const { return m_idata.template size<0>() == m_odata.template size<0>(); }
                        auto count() const { return m_idata.template size<0>(); }

                        auto idata(const tensor_size_t index)
                        {
                                assert(index >= 0 && index < count());
                                return m_idata.tensor(index);
                        }
                        auto idata(const tensor_size_t index) const
                        {
                                assert(index >= 0 && index < count());
                                return m_idata.tensor(index);
                        }

                        auto odata(const tensor_size_t index)
                        {
                                assert(index >= 0 && index < count());
                                return m_odata.tensor(index);
                        }
                        auto odata(const tensor_size_t index) const
                        {
                                assert(index >= 0 && index < count());
                                return m_odata.tensor(index);
                        }

                        titensor                m_idata;        ///< 4d inputs tensor: count x idims
                        totensor                m_odata;        ///< 4d outputs/targets tensor: count x odims
                };

                static size_t findex(const fold_t& fold)
                {
                        assert(fold.m_index < fsize());
                        return fold.m_index + static_cast<size_t>(fold.m_protocol) * fsize();
                }

                fold_data_t& fdata(const fold_t& fold) { return m_fdata[findex(fold)]; }
                const fold_data_t& fdata(const fold_t& fold) const { return m_fdata[findex(fold)]; }

                titensor& itensor(const fold_t& fold) { return fdata(fold).m_idata; }
                totensor& otensor(const fold_t& fold) { return fdata(fold).m_odata; }

                const titensor& itensor(const fold_t& fold) const { return fdata(fold).m_idata; }
                const totensor& otensor(const fold_t& fold) const { return fdata(fold).m_odata; }

                auto isample(const fold_t& fold, const tensor_size_t index) { return fdata(fold).idata(index); }
                auto osample(const fold_t& fold, const tensor_size_t index) { return fdata(fold).odata(index); }

                auto isample(const fold_t& fold, const tensor_size_t index) const { return fdata(fold).idata(index); }
                auto osample(const fold_t& fold, const tensor_size_t index) const { return fdata(fold).odata(index); }

        private:

                // attributes
                tensor3d_dims_t                 m_idims;        ///< input size
                tensor3d_dims_t                 m_odims;        ///< output size
                size_t                          m_fsize;        ///< number of folds
                mutable random_t<size_t>        m_frand;        ///< rng for fold assignment (training vs. validation vs. test)
                std::vector<fold_data_t>        m_fdata;        ///< samples / fold
        };

        template <typename titensor, typename totensor>
        mem_task_t::mem_task_t(
                const tensor3d_dims_t& idims,
                const tensor3d_dims_t& odims,
                const size_t fsize,
                const string_t& params = string_t()) :
                task_t(params),
                m_idims(idims), m_odims(odims),
                m_fsize(fsize), m_frand(1, 10)
        {
        }

        template <typename titensor, typename totensor>
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

        template <typename titensor, typename totensor>
        size_t mem_task_t<tchunk, tsample>::size() const
        {
                return  std::accumulate(m_samples.begin(), m_samples.end(), size_t(0),
                        [&] (const size_t count, const auto& samples) { return count + samples.second.size(); });
        }

        template <typename titensor, typename totensor>
        size_t mem_task_t<tchunk, tsample>::size(const fold_t& fold) const
        {
                const auto it = m_samples.find(fold);
                assert(it != m_samples.end());
                return it->second.size();
        }

        template <typename titensor, typename totensor>
        void mem_task_t<tchunk, tsample>::shuffle(const fold_t& fold) const
        {
                const auto it = m_samples.find(fold);
                assert(it != m_samples.end());

                std::random_device rd;
                std::minstd_rand g(rd());
                std::shuffle(it->second.begin(), it->second.end(), g);
        }

        template <typename titensor, typename totensor>
        sample_t mem_task_t<tchunk, tsample>::get(const fold_t& fold, const size_t index) const
        {
                const auto& sample = get_sample(fold, index);
                const auto& chunk = get_chunk(sample);
                return {sample.input(chunk), sample.target(), sample.label()};
        }

        template <typename titensor, typename totensor>
        size_t mem_task_t<tchunk, tsample>::hash(const fold_t& fold, const size_t index) const
        {
                const auto& fdata = get_fdata(fold);

                const auto& sample = get_sample(fold, index);
                const auto& hash = get_hash(sample);
                return sample.hash(hash);
        }
}
