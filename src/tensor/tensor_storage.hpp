#pragma once

#include "vector.hpp"
#include "matrix.hpp"
#include "tensor_index.hpp"

namespace tensor
{
        ///
        /// \brief 3+D tensor stored as ::dims() 2D planes of size ::rows() x ::cols()
        ///
        template
        <
                typename tstorage,      ///< data storage type (e.g. Eigen::Vector or mapped C-array)
                typename tdimensions
        >
        class tensor_storage_t
        {
        public:

                static_assert(tdimensions >= 3,
                        "cannot create tensors with fewer than 3 dimensions, use a vector or a matrix instead");

                using tsize = typename tstorage::Index;
                using tindex = typename tstorage::Index;
                using tscalar = typename tstorage::Scalar;

                using tdims = typename tensor_index_t<tindex, tdimensions>;

                // Eigen compatible
                using Index = tindex;
                using Scalar = tscalar;

                ///
                /// \brief constructor
                ///
                tensor_storate_t() = default;

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                tensor_storage_t(const tsizes... dims) :
                        m_dims(dims)
                {
                }

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                tensor_storage_t(const tsizes... dims, const tstorage& data) :
                        m_dims(dims),
                        m_data(data)
                {
                        assert(m_data.size() == m_dims.size());
                }

                ///
                /// \brief set all elements to zero
                ///
                void setZero()
                {
                        m_data.setZero();
                }

                ///
                /// \brief set all elements to constant
                ///
                void setConstant(const tscalar val)
                {
                        m_data.setConstant(val);
                }

                ///
                /// \brief dimensions
                ///
                tsize size() const { return m_data.size(); }
                template <int tdim>
                tsize dims() const { return m_dims.size<tdim>(); }
                tsize rows() const { return dims<tdimensions - 2>(); }
                tsize cols() const { return dims<tdimensions - 1>(); }
                tsize planeSize() const { return rows() * cols(); }

                ///
                /// \brief access the whole tensor as a vector (size() x 1)
                ///
                const tstorage& vector() const
                {
                        return m_data;
                }
                auto vector()
                {
                        return tensor::map_vector(data(), size());
                }

                ///
                /// \brief access the whole tensor as an array
                ///
                const tscalar* data() const
                {
                        return m_data.data();
                }
                tscalar* data()
                {
                        return m_data.data();
                }

                ///
                /// \brief access the 2D plane (i) as vector
                ///
                auto vector(const tindex i) const
                {
                        return tensor::map_vector(planeData(i), planeSize());
                }
                auto vector(const tindex i)
                {
                        return tensor::map_vector(planeData(i), planeSize());
                }

                ///
                /// \brief access the 2D plane (i) as matrix
                ///
                auto matrix(const tindex i) const
                {
                        return tensor::map_matrix(planeData(i), rows(), cols());
                }
                auto matrix(const tindex i)
                {
                        return tensor::map_matrix(planeData(i), rows(), cols());
                }

                ///
                /// \brief access the 2D plane (i) as an array
                ///
                const tscalar* planeData(const tindex i) const
                {
                        assert(i >= 0 && i < dims());
                        return data() + i * planeSize();
                }
                tscalar* planeData(const tindex i)
                {
                        assert(i >= 0 && i < dims());
                        return data() + i * planeSize();
                }

                ///
                /// \brief access an element of the tensor in the range [0, size())
                ///
                tscalar operator()(const tindex i) const
                {
                        assert(i >= 0 && i < size());
                        return m_data(i);
                }
                tscalar& operator()(const tindex i)
                {
                        assert(i >= 0 && i < size());
                        return m_data(i);
                }

        protected:

                // attributes
                tdims           m_dims;         ///< #dimensions
                tstorage        m_data;         ///< storage (1D vector)
        };
}
