#pragma once

#include "tensor_base.h"

namespace nano
{
        template <typename tstorage, std::size_t trank>
        struct tensor_array_t;

        ///
        /// \brief tensor mapping a non-constant array.
        ///
        template <typename tscalar, std::size_t trank>
        using tensor_map_t = tensor_array_t<tscalar*, trank>;

        ///
        /// \brief tensor mapping a constant array.
        ///
        template <typename tscalar, std::size_t trank>
        using tensor_const_map_t = tensor_array_t<const tscalar*, trank>;

        ///
        /// \brief map non-constant data to tensors
        ///
        template <typename tvalue_, std::size_t trank>
        auto map_tensor(tvalue_* data, const tensor_dims_t<trank>& dims)
        {
                using tvalue = typename std::remove_const<tvalue_>::type;
                return tensor_map_t<tvalue, trank>(data, dims);
        }

        ///
        /// \brief map constant data to tensors
        ///
        template <typename tvalue_, std::size_t trank>
        auto map_tensor(const tvalue_* data, const tensor_dims_t<trank>& dims)
        {
                using tvalue = typename std::remove_const<tvalue_>::type;
                return tensor_const_map_t<tvalue, trank>(data, dims);
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
        /// \brief
        ///
        template <typename ttensor, typename... tindices>
        auto subtensor(ttensor& tensor, const tindices... indices)
        {
                static_assert(sizeof...(indices) < ttensor::rank(), "invalid number of tensor dimensions");
                return map_tensor(tensor.data(indices...), nano::dims0(tensor.dims(), indices...));
        }

        ///
        /// \brief
        ///
        template <typename ttensor, typename... tsizes>
        auto reshape(ttensor& tensor, const tsizes... sizes)
        {
                assert(nano::size(nano::make_dims(sizes...)) == tensor.size());
                return map_tensor(tensor.data(), sizes...);
        }

        ///
        /// \brief tensor mapping a const or non-const 1D C-array.
        ///
        template <typename tstorage, std::size_t trank>
        struct tensor_array_t : public tensor_base_t<trank>
        {
                using tbase = tensor_base_t<trank>;

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
                tensor_array_t(tstorage pdata, const tsizes... dims) :
                        tbase(dims...),
                        m_data(pdata)
                {
                        assert(pdata != nullptr || this->size() == 0);
                }

                ///
                /// \brief set all elements to zero
                ///
                void zero() const { tbase::zero(array()); }
                void setZero() const { zero(); }

                ///
                /// \brief set all elements to a constant value
                ///
                void constant(const tscalar value) const { tbase::constant(array(), value); }
                void setConstant(const tscalar value) const { constant(value); }

                ///
                /// \brief set all elements to random values in the [min, max] range
                ///
                void random(const tscalar min, const tscalar max) const { tbase::random(array(), min, max); }
                void setRandom(const tscalar min, const tscalar max) const { random(min, max); }

                ///
                /// \brief access the tensor as a C-array
                ///
                auto data() const
                {
                        assert(m_data != nullptr);
                        return m_data;
                }

                template <typename... tindices>
                auto data(const tindices... indices) const { return tbase::data(data(), indices...); }

                ///
                /// \brief access the tensor as a vector
                ///
                auto vector() const { return map_vector(data(), this->size()); }

                template <typename... tindices>
                auto vector(const tindices... indices) const { return tbase::vector(data(), indices...); }

                ///
                /// \brief access the tensor as an array
                ///
                auto array() const { return vector().array(); }

                template <typename... tindices>
                auto array(const tindices... indices) const { return tbase::array(data(), indices...); }

                ///
                /// \brief access the tensor as a matrix
                ///
                template <typename... tindices>
                auto matrix(const tindices... indices) const { return tbase::matrix(data(), indices...); }

                ///
                /// \brief access the tensor as a (sub-)tensor
                ///
                template <typename... tindices>
                auto tensor(const tindices... indices) const
                {
                        return nano::subtensor(*this, indices...);
                }

                ///
                /// \brief access an element of the tensor
                ///
                treference operator()(const tensor_size_t index) const
                {
                        return m_data[index];
                }

                template <typename... tindices>
                treference operator()(const tensor_size_t index, const tindices... indices) const
                {
                        return m_data[this->offset(index, indices...)];
                }

                ///
                /// \brief reshape to a new tensor (with the same number of elements)
                ///
                template <typename... tsizes>
                auto reshape(const tsizes... sizes) const { return nano::reshape(*this, sizes...); }

        private:

                // attributes
                tstorage const  m_data;         ///< mapping a 1D C-array
        };
}
