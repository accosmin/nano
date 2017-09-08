#pragma once

#include "tensor_storage.h"

namespace nano
{
        ///
        /// \brief tensor that owns the allocated memory.
        ///
        template <typename tstorage_, std::size_t trank>
        struct tensor_t : public tensor_base_t<trank>
        {
                using tstorage = tensor_storage_t<tstorage_>;
                using tscalar = typename tstorage::tscalar;
                using treference = typename tstorage::treference;
                using tconst_reference = typename tstorage::tconst_reference;

                static constexpr bool resizable = typename tstorage::resizable;

                using tbase = tensor_base_t<trank>;
                using tdims = typename tbase::tdims;
                using Index = tensor_index_t;
                using Scalar = tscalar;

                ///
                /// \brief constructor
                ///
                tensor_t() = default;

                ///
                /// \brief constructors that resize the storage to match the given size
                ///
                template <typename = typename std::enable_if<resizable>::type, typename... tsizes>
                explicit tensor_t(const tsizes... dims) :
                        tbase(dims...),
                        m_storage(this->size())
                {
                }

                template <typename = typename std::enable_if<resizable>::type>
                explicit tensor_t(const tdims& dims) :
                        tbase(dims),
                        m_storage(this->size())
                {
                }

                ///
                /// \brief constructors that
                ///
                template <typename = typename std::enable_if<!resizable>::type>
                explicit tensor_t(const tstorage& storage, const tdims& dims) :
                        tbase(dims),
                        m_storage(storage)
                {
                }

                ///
                /// \brief copy constructor (todo)
                ///
                template <typename tstorage2>
                tensor_t(const tensor_t<tstorage2>& other);

                ///
                /// \brief assignment operator (todo)
                ///
                template <typename tstorage2>
                tensor_t& operator=(const tensor_t<tstorage2>& other);

                ///
                /// \brief resize to new dimensions
                ///
                template <typename = typename std::enable_if<resizable>::type, typename... tsizes>
                tensor_size_t resize(const tsizes... dims)
                {
                        return resize({{dims...}});
                }

                tensor_size_t resize(const tdims& dims)
                {
                        this->m_dims = dims;
                        this->m_storage.resize(this->size());
                        return this->size();
                }

                ///
                /// \brief set all elements to zero
                ///
                void zero() { tbase::zero(m_data); }
                void setZero() { zero(); }

                ///
                /// \brief set all elements to a constant value
                ///
                void constant(const tscalar value) { tbase::constant(m_data, value); }
                void setConstant(const tscalar value) { constant(value); }

                ///
                /// \brief set all elements to random values in the [min, max] range
                ///
                void random(const tscalar min, const tscalar max) { tbase::random(array(), min, max); }
                void setRandom(const tscalar min, const tscalar max) { random(min, max); }

                ///
                /// \brief access the tensor as a C-array
                ///
                auto data() const { return m_storage.data(); }
                auto data() { return m_storage.data(); }

                template <typename... tindices>
                auto data(const tindices... indices) const { return tbase::data(data(), indices...); }

                template <typename... tindices>
                auto data(const tindices... indices) { return tbase::data(data(), indices...); }

                ///
                /// \brief access the tensor as a vector
                ///
                auto vector() const { return map_vector(data(), this->size()); }
                auto vector() { return map_vector(data(), this->size()); }

                template <typename... tindices>
                auto vector(const tindices... indices) const { return tbase::vector(data(), indices...); }

                template <typename... tindices>
                auto vector(const tindices... indices) { return tbase::vector(data(), indices...); }

                ///
                /// \brief access the tensor as an array
                ///
                auto array() const { return vector().array(); }
                auto array() { return vector().array(); }

                template <typename... tindices>
                auto array(const tindices... indices) const { return tbase::array(data(), indices...); }

                template <typename... tindices>
                auto array(const tindices... indices) { return tbase::array(data(), indices...); }

                ///
                /// \brief access the tensor as a matrix
                ///
                template <typename... tindices>
                auto matrix(const tindices... indices) const { return tbase::matrix(data(), indices...); }

                template <typename... tindices>
                auto matrix(const tindices... indices) { return tbase::matrix(data(), indices...); }

                ///
                /// \brief access the tensor as a (sub-)tensor
                ///
                template <typename... tindices>
                auto tensor(const tindices... indices) const { return nano::subtensor(*this, indices...); }

                template <typename... tindices>
                auto tensor(const tindices... indices) { return nano::subtensor(*this, indices...); }

                ///
                /// \brief access an element of the tensor
                ///
                const tscalar& operator()(const tensor_size_t index) const { return m_data(index); }
                tscalar& operator()(const tensor_size_t index) { return m_data(index); }

                template <typename... tindices>
                const tscalar& operator()(const tensor_size_t index, const tindices... indices) const
                {
                        return m_data(this->offset(index, indices...));
                }

                template <typename... tindices>
                tscalar& operator()(const tensor_size_t index, const tindices... indices)
                {
                        return m_data(this->offset(index, indices...));
                }

                ///
                /// \brief reshape to a new tensor (with the same number of elements)
                ///
                template <typename... tsizes>
                auto reshape(const tsizes... sizes) const { return nano::reshape(*this, sizes...); }

                template <typename... tsizes>
                auto reshape(const tsizes... sizes) { return nano::reshape(*this, sizes...); }

        private:

                // attributes
                tstorage        m_storage;
        };
}
