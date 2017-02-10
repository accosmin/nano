#pragma once

#include "storage.h"

namespace tensor
{
        ///
        /// \brief 3+D tensor mapping an array of 2D planes.
        ///
        template
        <
                typename tmap,
                int tdimensions
        >
        struct tensor_map_t : public storage_t<tmap, tdimensions>
        {
                using tbase = storage_t<tmap, tdimensions>;
                using tdims = typename tbase::tdims;
                using tscalar = typename tbase::tscalar;

                // Eigen compatible
                using Index = index_t;
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
        /// \brief 3+D tensor stored as 2D planes.
        ///
        template
        <
                typename tscalar,
                int tdimensions,
                typename tvector = vector_t<tscalar>
        >
        struct tensor_t : public storage_t<tvector, tdimensions>
        {
                using tbase = storage_t<tvector, tdimensions>;
                using tdims = typename tbase::tdims;

                // Eigen compatible
                using Index = index_t;
                using Scalar = tscalar;

                ///
                /// \brief constructor
                ///
                tensor_t() = default;

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                explicit tensor_t(const tsizes... dims) :
                        tbase(dims...)
                {
                        this->m_data.resize(tensor::size(this->m_dims));
                }

                ///
                /// \brief constructor
                ///
                template <typename tmap>
                tensor_t(const tensor_map_t<tmap, tdimensions>& other) :
                        tensor_t(other.dims())
                {
                        this->m_data.resize(tensor::size(this->m_dims));
                        this->vector() = other.vector().template cast<tscalar>();
                }

                ///
                /// \brief resize to new dimensions
                ///
                template <typename... tsizes>
                index_t resize(const tsizes... dims)
                {
                        return resize({{dims...}});
                }

                ///
                /// \brief resize to new dimensions
                ///
                index_t resize(const tdims& dims)
                {
                        this->m_dims = dims;
                        this->m_data.resize(tensor::size(this->m_dims));
                        return this->size();
                }
        };

        ///
        /// \brief map non-constant data to tensors
        ///
        template <typename tvalue_, std::size_t tdimensions>
        auto map_tensor(tvalue_* data, const dims_t<tdimensions>& dims)
        {
                using tvalue = typename std::remove_const<tvalue_>::type;
                using tstorage = Eigen::Map<vector_t<tvalue>>;
                return tensor_map_t<tstorage, tdimensions>(tensor::map_vector(data, tensor::size(dims)), dims);
        }

        ///
        /// \brief map constant data to tensors
        ///
        template <typename tvalue_, std::size_t tdimensions>
        auto map_tensor(const tvalue_* data, const dims_t<tdimensions>& dims)
        {
                using tvalue = typename std::remove_const<tvalue_>::type;
                using tstorage = Eigen::Map<const vector_t<tvalue>>;
                return tensor_map_t<tstorage, tdimensions>(tensor::map_vector(data, tensor::size(dims)), dims);
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
