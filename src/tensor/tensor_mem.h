#pragma once

#include "tensor_map.h"

namespace nano
{
        ///
        /// \brief tensor that owns the allocated memory.
        ///
        template <typename tscalar, std::size_t trank>
        struct tensor_mem_t : public tensor_base_t<trank>
        {
                using tbase = tensor_base_t<trank>;

                using tvector = tensor_vector_t<tscalar>;
                using tdims = typename tbase::tdims;
                using Index = typename tbase::Index;
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
                        m_data.resize(this->size());
                }

                ///
                /// \brief constructor
                ///
                explicit tensor_mem_t(const tdims& dims) :
                        tbase(dims)
                {
                        m_data.resize(this->size());
                }

                ///
                /// \brief constructor
                ///
                template <typename tscalar2>
                tensor_mem_t(const tensor_array_t<tscalar2, trank>& other) :
                        tbase(other.dims())
                {
                        m_data = other.vector();
                }

                ///
                /// \brief resize to new dimensions
                ///
                template <typename... tsizes>
                tensor_size_t resize(const tsizes... dims)
                {
                        return resize({{dims...}});
                }

                ///
                /// \brief resize to new dimensions
                ///
                tensor_size_t resize(const tdims& dims)
                {
                        this->m_dims = dims;
                        this->m_data.resize(nano::size(this->m_dims));
                        return this->size();
                }

                ///
                /// \brief set all elements to zero
                ///
                void zero()
                {
                        m_data.setZero();
                }
                void setZero()
                {
                        zero();
                }

                ///
                /// \brief set all elements to a constant value
                ///
                void constant(const tscalar value)
                {
                        m_data.setConstant(value);
                }
                void setConstant(const tscalar value)
                {
                        constant(value);
                }

                ///
                /// \brief set all elements to random values in the [min, max] range
                ///
                void random(const tscalar min, const tscalar max)
                {
                        assert(min < max);
                        m_data.setRandom(); // [-1, +1]
                        array() = (array() + 1) * (max - min) / 2 + min;
                }
                void setRandom(const tscalar min, const tscalar max)
                {
                        random(min, max);
                }

                ///
                /// \brief access the tensor as a C-array
                ///
                auto data() const { return m_data.data(); }
                auto data() { return m_data.data(); }

                template <typename... tindices>
                auto data(const tindices... indices) const { return data() + this->offset(indices...); }

                template <typename... tindices>
                auto data(const tindices... indices) { return data() + this->offset(indices...); }

                ///
                /// \brief access the tensor as a vector
                ///
                auto vector() const { return map_vector(data(), this->size()); }
                auto vector() { return map_vector(data(), this->size()); }

                template <typename... tindices>
                auto vector(const tensor_size_t rows, const tindices... indices) const
                {
                        return this->mvector(data(), rows, indices...);
                }

                template <typename... tindices>
                auto vector(const tensor_size_t rows, const tindices... indices)
                {
                        return this->mvector(data(), rows, indices...);
                }

                ///
                /// \brief access the tensor as an array
                ///
                auto array() const { return vector().array(); }
                auto array() { return vector().array(); }

                template <typename... tindices>
                auto array(const tensor_size_t rows, const tindices... indices) const
                {
                        return this->marray(data(), rows, indices...);
                }

                template <typename... tindices>
                auto array(const tensor_size_t rows, const tindices... indices)
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

                template <typename... tindices>
                auto matrix(const tensor_size_t rows, const tensor_size_t cols, const tindices... indices)
                {
                        return this->mmatrix(data(), rows, cols, indices...);
                }

                ///
                /// \brief access an element of the tensor
                ///
                const tscalar& operator()(const tensor_size_t index) const
                {
                        return m_data(index);
                }
                tscalar& operator()(const tensor_size_t index)
                {
                        return m_data(index);
                }

                template <typename... tindices>
                const tscalar& operator()(const tensor_size_t index, const tindices... indices) const
                {
                        return this->operator()(this->offset(index, indices...));
                }

                template <typename... tindices>
                tscalar& operator()(const tensor_size_t index, const tindices... indices)
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

                template <typename... tsizes>
                auto reshape(const tsizes... sizes)
                {
                        assert(nano::size(nano::make_dims(sizes...)) == this->size());
                        return map_tensor(data(), sizes...);
                }

        private:

                // attributes
                tvector         m_data;         ///< tensor stored as 1D vector
        };
}
