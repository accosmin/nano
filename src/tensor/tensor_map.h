#pragma once

#include "index.h"
#include "vector.h"
#include "matrix.h"

namespace nano
{
        template <typename tstorage, std::size_t tdimensions>
        struct tensor_array_t;

        ///
        /// \brief tensor mapping a non-constant array.
        ///
        template <typename tscalar, std::size_t tdimensions>
        using tensor_map_t = tensor_array_t<tscalar*, tdimensions>;

        ///
        /// \brief tensor mapping a constant array.
        ///
        template <typename tscalar, std::size_t tdimensions>
        using tensor_const_map_t = tensor_array_t<const tscalar*, tdimensions>;

        ///
        /// \brief map non-constant data to tensors
        ///
        template <typename tvalue_, std::size_t tdimensions>
        auto map_tensor(tvalue_* data, const tensor_dims_t<tdimensions>& dims)
        {
                using tvalue = typename std::remove_const<tvalue_>::type;
                return tensor_map_t<tvalue, tdimensions>(data, dims);
        }

        ///
        /// \brief map constant data to tensors
        ///
        template <typename tvalue_, std::size_t tdimensions>
        auto map_tensor(const tvalue_* data, const tensor_dims_t<tdimensions>& dims)
        {
                using tvalue = typename std::remove_const<tvalue_>::type;
                return tensor_const_map_t<tvalue, tdimensions>(data, dims);
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

        ///
        /// \brief tensor mapping a const or non-const 1D C-array.
        ///
        template <typename tstorage, std::size_t tdimensions>
        struct tensor_array_t : public tensor_indexer_t<tdimensions>
        {
                using tbase = tensor_indexer_t<tdimensions>;

                using tscalar = typename std::remove_pointer<tstorage>::type;
                using treference = typename std::conditional<std::is_const<tstorage>::value, const tscalar&, tscalar&>::type;
                using tdims = typename tbase::tdims;
                using Index = typename tbase::Index;
                using Scalar = tscalar;

                ///
                /// \brief constructor
                ///
                tensor_array_t() :
                        m_data(nullptr)
                {
                }

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                tensor_array_t(tstorage data, const tsizes... dims) :
                        tbase(dims...),
                        m_data(data)
                {
                        assert(data != nullptr || this->size() == 0);
                }

                ///
                /// \brief access the whole tensor as a vector (size() x 1)
                ///
                auto vector() const
                {
                        return map_vector(data(), this->size());
                }

                ///
                /// \brief access the whole tensor as an array (size() x 1)
                ///
                auto array() const
                {
                        return vector().array();
                }

                ///
                /// \brief access the whole tensor as a C-array
                ///
                auto data() const
                {
                        assert(m_data != nullptr);
                        return m_data;
                }

                ///
                /// \brief access the 2D plane (indices...) as a vector
                ///
                template <typename... tindices>
                auto vector(const tindices... indices) const
                {
                        return map_vector(planeData(indices...), this->planeSize());
                }

                ///
                /// \brief access the 2D plane (indices...) as an array
                ///
                template <typename... tindices>
                auto array(const tindices... indices) const
                {
                        return vector(indices...).array();
                }

                ///
                /// \brief access the 2D plane (indices...) as matrix
                ///
                template <typename... tindices>
                auto matrix(const tindices... indices) const
                {
                        return map_matrix(planeData(indices...), this->rows(), this->cols());
                }

                ///
                /// \brief access the 2D plane (indices...) as a C-array
                ///
                template <typename... tindices>
                auto planeData(const tindices... indices) const
                {
                        static_assert(sizeof...(indices) == tdimensions - 2, "method not available");
                        return data() + nano::index(this->dims(), indices..., 0, 0);
                }

                ///
                /// \brief access an element of the tensor
                ///
                treference operator()(const tensor_size_t index) const
                {
                        return m_data[index];
                }

                ///
                /// \brief access an element of the tensor
                ///
                template <typename... tindices>
                treference operator()(const tensor_size_t index, const tindices... indices) const
                {
                        return m_data[nano::index(this->dims(), index, indices...)];
                }

        private:

                // attributes
                tstorage const  m_data;         ///< mapping a 1D C-array
        };
}
