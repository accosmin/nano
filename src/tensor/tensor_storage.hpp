#pragma once

#include "vector.hpp"
#include "matrix.hpp"
#include "tensor_index.hpp"

namespace tensor
{
        ///
        /// \brief 3D tensor stored as ::dims() 2D planes of size ::rows() x ::cols()
        ///
        template
        <
                typename tstorage        ///< data storage type (e.g. Eigen::Vector or mapped C-array)
        >
        class tensor_storage_t
        {
        public:

                using tsize = typename tstorage::Index;
                using tindex = typename tstorage::Index;
                using tscalar = typename tstorage::Scalar;

                // Eigen compatible
                using Index = tindex;
                using Scalar = tscalar;

                ///
                /// \brief constructor
                ///
                tensor_storage_t(const tsize dims, const tsize rows, const tsize cols) :
                        m_dims(dims),
                        m_rows(rows),
                        m_cols(cols)
                {
                }

                ///
                /// \brief constructor
                ///
                tensor_storage_t(const tsize dims, const tsize rows, const tsize cols, const tstorage& data) :
                        m_dims(dims),
                        m_rows(rows),
                        m_cols(cols),
                        m_data(data)
                {
                        assert(m_data.size() == dims * rows * cols);
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
                tsize dims() const { return m_dims; }
                tsize rows() const { return m_rows; }
                tsize cols() const { return m_cols; }
                tsize planeSize() const { return rows() * cols(); }

                ///
                /// \brief access the whole tensor as a vector (size() x 1)
                ///
                const tstorage& vector() const { return m_data; }
                auto vector() { return tensor::map_vector(data(), size()); }

                ///
                /// \brief access the whole tensor as an array
                ///
                const tscalar* data() const { return m_data.data(); }
                tscalar* data() { return m_data.data(); }

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
                tsize           m_dims;         ///< #dimensions
                tsize           m_rows;         ///< #rows (for each dimension)
                tsize           m_cols;         ///< #cols (for each dimension)
                tstorage        m_data;         ///< storage (1D vector)
        };
}
