#include "task_mem_tensor.h"
#include "math/random.hpp"

namespace nano
{
        mem_tensor_task_t::mem_tensor_task_t(
                const tensor_size_t idims, const tensor_size_t irows, const tensor_size_t icols,
                const tensor_size_t osize) :
                m_idims(idims), m_irows(irows), m_icols(icols), m_osize(osize)
        {
        }

        void mem_tensor_task_t::clear()
        {
                m_data.clear();
                m_data.shrink_to_fit();
        }

        void mem_tensor_task_t::clear(const fold_t& fold)
        {
                m_data.erase(fold);
        }

        void mem_tensor_task_t::reserve(const fold_t& fold, const size_t count)
        {
                auto& data = m_data[fold];
                data.clear();
                data.reserve(count);
        }

        void mem_tensor_task_t::push_back(const fold_t& fold, const tensor3d_t& input, const target_t& target)
        {
                auto& data = m_data[fold];
                data.emplace_back(input, target);
        }

        size_t mem_tensor_task_t::n_folds() const
        {
                size_t max_fold = 0;
                for (const auto& data : m_data)
                {
                        max_fold = std::max(max_fold, data.first.m_index);
                }

                return max_fold + 1;
        }

        size_t mem_tensor_task_t::n_samples() const
        {
                size_t count = 0;
                for (const auto& data : m_data)
                {
                        count += data.second.size();
                }

                return count;
        }

        size_t mem_tensor_task_t::n_samples(const fold_t& fold) const
        {
                const auto it = m_data.find(fold);
                assert(it != m_data.end());
                return it->second.size();
        }

        void mem_tensor_task_t::shuffle(const fold_t& fold)
        {
                const auto it = m_data.find(fold);
                assert(it != m_data.end());

                random_t<size_t> rng(0, it->second.size());
                std::shuffle(it->second.begin(), it->second.end(), rng);
        }

        tensor3d_t mem_tensor_task_t::input(const fold_t& fold, const size_t index) const
        {
                const auto it = m_data.find(fold);
                assert(it != m_data.end());
                assert(index < it->second.size());
                return it->second[index].m_input;
        }

        target_t mem_tensor_task_t::target(const fold_t& fold, const size_t index) const
        {
                const auto it = m_data.find(fold);
                assert(it != m_data.end());
                assert(index < it->second.size());
                return it->second[index].m_target;
        }
}
