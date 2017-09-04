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
                void zero() const
                {
                        this->array().setZero();
                }
                void setZero() const
                {
                        zero();
                }

                ///
                /// \brief set all elements to a constant value
                ///
                void constant(const tscalar value) const
                {
                        this->array().setConstant(value);
                }
                void setConstant(const tscalar value) const
                {
                        constant(value);
                }

                ///
                /// \brief set all elements to random values in the [min, max] range
                ///
                void random(const tscalar min, const tscalar max) const
                {
                        assert(min < max);
                        array().setRandom(); // [-1, +1]
                        array() = (array() + 1) * (max - min) / 2 + min;
                }
                void setRandom(const tscalar min, const tscalar max) const
                {
                        random(min, max);
                }

                ///
                /// \brief access the tensor as a C-array
                ///
                auto data() const
                {
                        assert(m_data != nullptr);
                        return m_data;
                }

                template <typename... tindices>
                auto data(const tindices... indices) const
                {
                        return data() + this->offset(indices...);
                }

                ///
                /// \brief access the tensor as a vector
                ///
                auto vector() const { return map_vector(data(), this->size()); }

                template <typename... tindices>
                auto vector(const tensor_size_t rows, const tindices... indices) const
                {
                        return this->mvector(data(), rows, indices...);
                }

                ///
                /// \brief access the tensor as an array
                ///
                auto array() const { return vector().array(); }

                template <typename... tindices>
                auto array(const tensor_size_t rows, const tindices... indices) const
                {
                        return this->marray(data(), rows, indices...);
                }

                ///
                /// \brief access the tensor as a matrix
                ///
                template <typename... tindices>
                auto matrix(const tensor_size_t rows, const tensor_size_t cols, const tindices... indices) const
                {
                        return this->mmatrix(data(), rows, cols, indices...);
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
                        return this->operator()(this->offset(index, indices...));
                }

                ///
                /// \brief reshape to a new tensor (with the same number of elements)
                ///
                template <typename... tsizes>
                auto reshape(const tsizes... sizes) const
                {
                        assert(nano::size(nano::make_dims(sizes...)) == this->size());
                        return map_tensor(data(), sizes...);
                }

        private:

                // attributes
                tstorage const  m_data;         ///< mapping a 1D C-array
        };
}
