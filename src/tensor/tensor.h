#pragma once

#include "storage.h"

namespace nano
{
        ///
        /// \brief 3+D tensor mapping an array of 2D planes.
        ///
        template <typename tmap, std::size_t tdimensions>
        struct tensor_map_t : public tensor_storage_t<tmap, tdimensions>
        {
                using tbase = tensor_storage_t<tmap, tdimensions>;
                using tdims = typename tbase::tdims;
                using tscalar = typename tbase::tscalar;

                // Eigen compatible
                using Index = tensor_index_t;
                using Scalar = tscalar;

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                tensor_map_t(const tmap& map, const tsizes... dims) :
                        tbase(map, dims...)
                {
                }
        };

        ///
        /// \brief 3+D tensor stored as 2D planes that owns the allocated memory.
        ///
        template <typename tvector, std::size_t tdimensions>
        struct tensor_mem_t : public tensor_storage_t<tvector, tdimensions>
        {
                using tbase = tensor_storage_t<tvector, tdimensions>;
                using tdims = typename tbase::tdims;
                using tscalar = typename tbase::tscalar;

                // Eigen compatible
                using Index = tensor_index_t;
                using Scalar = tscalar;

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
                template <typename tmap>
                tensor_mem_t(const tensor_map_t<tmap, tdimensions>& other) :
                        tensor_mem_t(other.dims())
                {
                        this->vector() = other.vector().template cast<tscalar>();
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

        ///
        /// \brief map non-constant data to tensors
        ///
        template <typename tvalue_, std::size_t tdimensions>
        auto map_tensor(tvalue_* data, const tensor_dims_t<tdimensions>& dims)
        {
                using tvalue = typename std::remove_const<tvalue_>::type;
                using tstorage = Eigen::Map<tensor_vector_t<tvalue>>;
                return tensor_map_t<tstorage, tdimensions>(nano::map_vector(data, nano::size(dims)), dims);
        }

        ///
        /// \brief map constant data to tensors
        ///
        template <typename tvalue_, std::size_t tdimensions>
        auto map_tensor(const tvalue_* data, const tensor_dims_t<tdimensions>& dims)
        {
                using tvalue = typename std::remove_const<tvalue_>::type;
                using tstorage = Eigen::Map<const tensor_vector_t<tvalue>>;
                return tensor_map_t<tstorage, tdimensions>(nano::map_vector(data, nano::size(dims)), dims);
        }

        ///
        /// \brief map non-constant data to tensors
        ///
        template <typename tvalue_, typename... tsizes>
        auto map_tensor(tvalue_* data, const tsizes... dims)
        {
                return map_tensor(data, make_dims(dims...));
        }

        ///
        /// \brief map constant data to tensors
        ///
        template <typename tvalue_, typename... tsizes>
        auto map_tensor(const tvalue_* data, const tsizes... dims)
        {
                return map_tensor(data, make_dims(dims...));
        }
}
