#pragma once

#include "tensor_map.h"

namespace nano
{
        ///
        /// \brief 3+D tensor stored as 2D planes that owns the allocated memory.
        ///
        template <typename tscalar, std::size_t tdimensions, typename tvector = tensor_vector_t<tscalar>>
        struct tensor_mem_t : public tensor_storage_t<tvector, tdimensions>
        {
                using tbase = tensor_storage_t<tvector, tdimensions>;

                using tdims = typename tbase::tdims;
                using Index = typename tbase::Index;
                using Scalar = typename tbase::Scalar;

                ///
                /// \brief constructor
                ///
                tensor_mem_t() = default;

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                explicit tensor_mem_t(const tsizes... dims) :
                        tbase(dims...)
                {
                        this->m_data.resize(nano::size(this->m_dims));
                }

                ///
                /// \brief constructor
                ///
                tensor_mem_t(const tensor_map_t<tscalar, tdimensions>& other) :
                        tensor_mem_t(other.dims())
                {
                        this->vector() = other.vector();
                }

                ///
                /// \brief constructor
                ///
                tensor_mem_t(const tensor_const_map_t<tscalar, tdimensions>& other) :
                        tensor_mem_t(other.dims())
                {
                        this->vector() = other.vector();
                }

                ///
                /// \brief resize to new dimensions
                ///
                template <typename... tsizes>
                tensor_index_t resize(const tsizes... dims)
                {
                        return resize({{dims...}});
                }

                ///
                /// \brief resize to new dimensions
                ///
                tensor_index_t resize(const tdims& dims)
                {
                        this->m_dims = dims;
                        this->m_data.resize(nano::size(this->m_dims));
                        return this->size();
                }
        };
}
