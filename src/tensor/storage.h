#pragma once

#include "index.h"
#include "vector.h"
#include "matrix.h"

namespace tensor
{
        ///
        /// \brief 3+D tensor stored as ::dims() 2D planes of size ::rows() x ::cols()
        ///
        template
        <
                typename tstorage,      ///< data storage type (e.g. Eigen::Vector or mapped C-array)
                int tdimensions
        >
        class storage_t
        {
        public:

                static_assert(tdimensions >= 3,
                        "cannot create tensors with fewer than 3 dimensions, use a vector or a matrix instead");

                using tscalar = typename tstorage::Scalar;
                using tdims = dims_t<tdimensions>;

                // Eigen compatible
                using Index = index_t;
                using Scalar = tscalar;

                ///
                /// \brief constructor
                ///
                storage_t() = default;

                ///
                /// \brief constructor
                ///
                explicit storage_t(const tdims& dims) :
                        m_dims(dims)
                {
                }

                ///
                /// \brief constructor
                ///
                template <typename... index_ts>
                explicit storage_t(const index_ts... dims) :
                        m_dims({{dims...}})
                {
                }

                ///
                /// \brief constructor
                ///
                template <typename... index_ts>
                storage_t(const tstorage& data, const index_ts... dims) :
                        m_dims({{dims...}}),
                        m_data(data)
                {
                        assert(m_data.size() == tensor::size(m_dims));
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
                index_t size() const { assert(m_data.size() == tensor::size(m_dims)); return m_data.size(); }
                template <int idim>
                index_t size() const { return std::get<idim>(m_dims); }
                index_t rows() const { return size<tdimensions - 2>(); }
                index_t cols() const { return size<tdimensions - 1>(); }
                index_t planeSize() const { return rows() * cols(); }
                const tdims& dims() const { return m_dims; }
                auto dimensionality() const { return tdimensions; }

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
                /// \brief access the 2D plane (indices...) as vector
                ///
                template <typename... tindices>
                auto vector(const tindices... indices) const
                {
                        return tensor::map_vector(planeData(indices...), planeSize());
                }
                template <typename... tindices>
                auto vector(const tindices... indices)
                {
                        return tensor::map_vector(planeData(indices...), planeSize());
                }

                ///
                /// \brief access the 2D plane (indices...) as matrix
                ///
                template <typename... tindices>
                auto matrix(const tindices... indices) const
                {
                        return tensor::map_matrix(planeData(indices...), rows(), cols());
                }
                template <typename... tindices>
                auto matrix(const tindices... indices)
                {
                        return tensor::map_matrix(planeData(indices...), rows(), cols());
                }

                ///
                /// \brief access the 2D plane (indices...) as an array
                ///
                template <typename... tindices>
                const tscalar* planeData(const tindices... indices) const
                {
                        static_assert(sizeof...(indices) == tdimensions - 2,
                                "wrong number of tensor dimensions to access a 2D plane");
                        return data() + tensor::index(m_dims, indices..., 0, 0);
                }
                template <typename... tindices>
                tscalar* planeData(const tindices... indices)
                {
                        static_assert(sizeof...(indices) == tdimensions - 2,
                                "wrong number of tensor dimensions to access a 2D plane");
                        return data() + tensor::index(m_dims, indices..., 0, 0);
                }

                ///
                /// \brief access an element of the tensor
                ///
                tscalar operator()(const index_t index) const
                {
                        return m_data(index);
                }
                tscalar& operator()(const index_t index)
                {
                        return m_data(index);
                }

                ///
                /// \brief access an element of the tensor
                ///
                template <typename... tindices>
                tscalar operator()(const tindices... indices) const
                {
                        return m_data(tensor::index(m_dims, indices...));
                }
                template <typename... tindices>
                tscalar& operator()(const tindices... indices)
                {
                        return m_data(tensor::index(m_dims, indices...));
                }

        protected:

                // attributes
                tdims           m_dims;         ///< dimensions
                tstorage        m_data;         ///< storage (1D vector)
        };
}
