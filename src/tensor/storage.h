#pragma once

#include "index.h"
#include "vector.h"
#include "matrix.h"

namespace nano
{
        ///
        /// \brief constant storage for tensors.
        ///
        template <typename tstorage, std::size_t tdimensions>
        struct tensor_const_storage_t
        {
                static_assert(tdimensions >= 1, "cannot create tensors with fewer than one dimension");

                using tscalar = typename tstorage::Scalar;
                using tdims = tensor_dims_t<tdimensions>;

                // Eigen compatible
                using Index = tensor_index_t;
                using Scalar = tscalar;

                ///
                /// \brief constructor
                ///
                tensor_const_storage_t()
                {
                        m_dims.fill(0);
                }

                ///
                /// \brief constructor
                ///
                explicit tensor_const_storage_t(const tdims& dims) :
                        m_dims(dims)
                {
                }

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                explicit tensor_const_storage_t(const tsizes... dims) :
                        m_dims({dims...})
                {
                }

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                tensor_const_storage_t(const tstorage& data, const tsizes... dims) :
                        m_dims({dims...}),
                        m_data(data)
                {
                        assert(m_data.size() == nano::size(m_dims));
                }

                ///
                /// \brief dimensions
                ///
                tensor_index_t size() const { assert(m_data.size() == nano::size(m_dims)); return m_data.size(); }
                template <int idim>
                tensor_index_t size() const { return std::get<idim>(m_dims); }
                tensor_index_t rows() const { return nano::rows(m_dims); }
                tensor_index_t cols() const { return nano::cols(m_dims); }
                tensor_index_t planeSize() const { return nano::planeSize(m_dims); }
                const tdims& dims() const { return m_dims; }
                auto dimensionality() const { return tdimensions; }

                ///
                /// \brief access the whole tensor as a vector (size() x 1)
                ///
                auto vector() const
                {
                        return nano::map_vector(data(), size());
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
                const tscalar* data() const
                {
                        return m_data.data();
                }

                ///
                /// \brief access the 2D plane (indices...) as a vector
                ///
                template <typename... tindices>
                auto vector(const tindices... indices) const
                {
                        return nano::map_vector(planeData(indices...), planeSize());
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
                        static_assert(tdimensions >= 3, "method not available");
                        return nano::map_matrix(planeData(indices...), rows(), cols());
                }

                ///
                /// \brief access the 2D plane (indices...) as a C-array
                ///
                template <typename... tindices>
                const tscalar* planeData(const tindices... indices) const
                {
                        static_assert(tdimensions >= 3, "method not available");
                        static_assert(sizeof...(indices) == tdimensions - 2, "method not available");
                        return data() + nano::index(m_dims, indices..., 0, 0);
                }

                ///
                /// \brief access an element of the tensor
                ///
                const tscalar& operator()(const tensor_index_t index) const
                {
                        return m_data(index);
                }

                ///
                /// \brief access an element of the tensor
                ///
                template <typename... tindices>
                const tscalar& operator()(const tensor_index_t index, const tindices... indices) const
                {
                        return m_data(nano::index(m_dims, index, indices...));
                }

        protected:

                // attributes
                tdims           m_dims;         ///< dimensions
                tstorage        m_data;         ///< storage (1D vector)
        };

        ///
        /// \brief non-constant storage for tenors.
        ///
        template <typename tstorage, std::size_t tdimensions>
        struct tensor_storage_t : public tensor_const_storage_t<tstorage, tdimensions>
        {
                static_assert(tdimensions >= 1, "cannot create tensors with fewer than one dimension");

                using tbase = tensor_const_storage_t<tstorage, tdimensions>;

                using tscalar = typename tbase::tscalar;
                using tdims = typename tbase::tdims;
                using Index = typename tbase::Index;
                using Scalar = typename tbase::Scalar;

                using tbase::data;
                using tbase::array;
                using tbase::vector;
                using tbase::matrix;
                using tbase::planeData;
                using tbase::operator();

                using tbase::size;
                using tbase::rows;
                using tbase::cols;
                using tbase::dims;
                using tbase::planeSize;
                using tbase::dimensionality;

                ///
                /// \brief constructor
                ///
                tensor_storage_t() : tbase() {}

                ///
                /// \brief constructor
                ///
                explicit tensor_storage_t(const tdims& dims) : tbase(dims) {}

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                explicit tensor_storage_t(const tsizes... dims) : tbase(dims...) {}

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                tensor_storage_t(const tstorage& data, const tsizes... dims) : tbase(data, dims...) {}

                ///
                /// \brief set all elements to zero
                ///
                void setZero()
                {
                        this->m_data.setZero();
                }

                ///
                /// \brief set all elements to constant
                ///
                void setConstant(const tscalar val)
                {
                        this->m_data.setConstant(val);
                }

                ///
                /// \brief set all elements to random values in the [-1, +1] interval
                ///
                void setRandom()
                {
                        this->m_data.setRandom();
                }

                ///
                /// \brief access the whole tensor as a vector (size() x 1)
                ///
                auto vector()
                {
                        return nano::map_vector(data(), this->size());
                }

                ///
                /// \brief access the whole tensor as a vector (size() x 1)
                ///
                auto array()
                {
                        return vector().array();
                }

                ///
                /// \brief access the whole tensor as a C-array
                ///
                tscalar* data()
                {
                        return this->m_data.data();
                }

                ///
                /// \brief access the 2D plane (indices...) as a vector
                ///
                template <typename... tindices>
                auto vector(const tindices... indices)
                {
                        return nano::map_vector(planeData(indices...), this->planeSize());
                }

                ///
                /// \brief access the 2D plane (indices...) as an array
                ///
                template <typename... tindices>
                auto array(const tindices... indices)
                {
                        return vector(indices...).array();
                }

                ///
                /// \brief access the 2D plane (indices...) as a matrix
                ///
                template <typename... tindices>
                auto matrix(const tindices... indices)
                {
                        static_assert(tdimensions >= 3, "method not available");
                        return nano::map_matrix(planeData(indices...), this->rows(), this->cols());
                }

                ///
                /// \brief access the 2D plane (indices...) as a C-array
                ///
                template <typename... tindices>
                tscalar* planeData(const tindices... indices)
                {
                        static_assert(tdimensions >= 3, "method not available");
                        static_assert(sizeof...(indices) == tdimensions - 2, "method not available");
                        return data() + nano::index(this->dims(), indices..., 0, 0);
                }

                ///
                /// \brief access an element of the tensor
                ///
                tscalar& operator()(const tensor_index_t index)
                {
                        return this->m_data(index);
                }

                ///
                /// \brief access an element of the tensor
                ///
                template <typename... tindices>
                tscalar& operator()(const tensor_index_t index, const tindices... indices)
                {
                        return operator()(nano::index(this->dims(), index, indices...));
                }
        };
}
