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
                int tdimensions
        >
        class tensor_storage_t
        {
        public:

                static_assert(tdimensions >= 3,
                        "cannot create tensors with fewer than 3 dimensions, use a vector or a matrix instead");

                using tsize = typename tstorage::Index;
                using tindex = typename tstorage::Index;
                using tscalar = typename tstorage::Scalar;
                using tdims = tensor_index_t<tindex, tdimensions>;

                // Eigen compatible
                using Index = tindex;
                using Scalar = tscalar;

                ///
                /// \brief constructor
                ///
                tensor_storage_t() = default;

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                tensor_storage_t(const tsizes... dims) :
                        m_dims(dims...)
                {
                }

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                tensor_storage_t(const tsizes... dims, const tstorage& data) :
                        m_dims(dims...),
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
                template <int idim>
                tsize size() const { return m_dims.template dims<idim>(); }
                tsize rows() const { return size<tdimensions - 2>(); }
                tsize cols() const { return size<tdimensions - 1>(); }
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
                        return data() + m_dims(indices..., 0, 0);
                }
                template <typename... tindices>
                tscalar* planeData(const tindices... indices)
                {
                        static_assert(sizeof...(indices) == tdimensions - 2,
                                "wrong number of tensor dimensions to access a 2D plane");
                        return data() + m_dims(indices..., 0, 0);
                }

                ///
                /// \brief access an element of the tensor
                ///
                template <typename... tindices>
                tscalar operator()(const tindices... indices) const
                {
                        return m_data(m_dims(indices...));
                }
                template <typename... tindices>
                tscalar& operator()(const tindices... indices)
                {
                        return m_data(m_dims(indices...));
                }

        protected:

                // attributes
                tdims           m_dims;         ///< dimensions
                tstorage        m_data;         ///< storage (1D vector)
        };
}
