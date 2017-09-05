#pragma once

#include "vector.h"
#include "matrix.h"
#include "tensor_index.h"

namespace nano
{
        ///
        /// \brief base object for allocated or mapped tensors with support for indexing operations.
        ///
        template <std::size_t trank>
        struct tensor_base_t
        {
                static_assert(trank >= 1, "cannot create tensors with fewer than one dimension");

                using tdims = tensor_dims_t<trank>;
                using Index = tensor_size_t;

                ///
                /// \brief constructor
                ///
                tensor_base_t()
                {
                        m_dims.fill(0);
                }

                ///
                /// \brief constructor
                ///
                explicit tensor_base_t(const tdims& dims) :
                        m_dims(dims)
                {
                }

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                explicit tensor_base_t(const tsizes... dims) :
                        m_dims({dims...})
                {
                }

                ///
                /// \brief number of dimensions (aka the rank of the tensor)
                ///
                static auto rank() { return trank; }

                ///
                /// \brief list of dimensions
                ///
                const auto& dims() const { return m_dims; }

                ///
                /// \brief total number of elements
                ///
                auto size() const { return nano::size(m_dims); }

                ///
                /// \brief number of elements for the given dimension
                ///
                template <int idim>
                auto size() const { return std::get<idim>(m_dims); }

                ///
                /// \brief interpret the last two dimensions as rows/columns
                /// NB: e.g. images represented as 3D tensors (color plane, rows, columns)
                /// NB: e.g. ML minibatches represented as 4D tensors (sample, feature plane, rows, columns)
                ///
                auto rows() const { static_assert(trank >= 3, ""); return size<trank - 2>(); }
                auto cols() const { static_assert(trank >= 3, ""); return size<trank - 1>(); }

                ///
                /// \brief compute the linearized index from the list of offsets
                ///
                template <typename... tindices>
                auto offset(const tindices... offsets) const
                {
                        return nano::index(dims(), offsets...);
                }

        protected:

                template <typename tdata, typename... tindices>
                auto vector(tdata data, const tindices... indices) const
                {
                        static_assert(sizeof...(indices) == trank - 1, "invalid number of tensor dimensions");
                        return map_vector(data + offset(indices..., 0), cols());
                }

                template <typename tdata, typename... tindices>
                auto array(tdata data, const tindices... indices) const
                {
                        static_assert(sizeof...(indices) == trank - 1, "invalid number of tensor dimensions");
                        return vector(data, indices...).array();
                }

                template <typename tdata, typename... tindices>
                auto matrix(tdata data, const tindices... indices) const
                {
                        static_assert(sizeof...(indices) == trank - 2, "invalid number of tensor dimensions");
                        return map_matrix(data + offset(indices..., 0, 0), rows(), cols());
                }

                template <typename tdata, typename... tindices>
                auto tensor(tdata data, const tindices... indices) const
                {
                        static_assert(sizeof...(indices) > 0, "invalid number of tensor dimensions");
                        static_assert(sizeof...(indices) > trank, "invalid number of tensor dimensions");
                        // todo: generic tensor mapping
                        (void)data;
                }

                // attributes
                tdims           m_dims;         ///< dimensions
        };
}
