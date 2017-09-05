#pragma once

#include <array>
#include <cassert>
#include <ostream>
#include "vector.h"
#include "matrix.h"

namespace nano
{
        using tensor_size_t = Eigen::Index;

        ///
        /// \brief dimensions of a multi-dimensional tensor.
        ///
        template <std::size_t trank>
        using tensor_dims_t = std::array<tensor_size_t, trank>;

        ///
        /// \brief create a dimension structure for a tensor.
        ///
        template <typename... tsizes>
        auto make_dims(const tsizes... sizes)
        {
                return tensor_dims_t<sizeof...(sizes)>({{sizes...}});
        }

        namespace detail
        {
                template <std::size_t trank>
                tensor_size_t product(const tensor_dims_t<trank>& dims, const std::size_t idim)
                {
                        return (idim >= trank) ? 1 : dims[idim] * product(dims, idim + 1);
                }

                template <std::size_t trank>
                tensor_size_t get_index(const tensor_dims_t<trank>&, const std::size_t, const tensor_size_t index)
                {
                        return index;
                }

                template <std::size_t trank, typename... tindices>
                tensor_size_t get_index(const tensor_dims_t<trank>& dims, const std::size_t idim,
                        const tensor_size_t index, const tindices... indices)
                {
                        assert(index >= 0 && index < dims[idim]);
                        return index * product(dims, idim + 1) + get_index(dims, idim + 1, indices...);
                }

                template <std::size_t trank>
                tensor_size_t get_index_fill0(const tensor_dims_t<trank>&, const std::size_t)
                {
                        return 0;
                }

                template <std::size_t trank, typename... tindices>
                tensor_size_t get_index_fill0(const tensor_dims_t<trank>& dims, const std::size_t idim,
                        const tensor_size_t index, const tindices... indices)
                {
                        assert(index >= 0 && index < dims[idim]);
                        return index * product(dims, idim + 1) + get_index_fill0(dims, idim + 1, indices...);
                }
        }

        ///
        /// \brief index a multi-dimensional tensor.
        ///
        template <std::size_t trank, typename... tindices>
        tensor_size_t index(const tensor_dims_t<trank>& dims, const tindices... indices)
        {
                static_assert(trank >= 1, "invalid number of tensor dimensions");
                static_assert(sizeof...(indices) == trank, "invalid number of tensor indices");
                return detail::get_index(dims, 0, indices...);
        }

        ///
        /// \brief index a multi-dimensional tensor (assuming the last dimensions that are ignored are zero).
        ///
        template <std::size_t trank, typename... tindices>
        tensor_size_t index_fill0(const tensor_dims_t<trank>& dims, const tindices... indices)
        {
                static_assert(trank >= 1, "invalid number of tensor dimensions");
                static_assert(sizeof...(indices) <= trank, "invalid number of tensor indices");
                return detail::get_index_fill0(dims, 0, indices...);
        }

        ///
        /// \brief size of a multi-dimensional tensor (#elements).
        ///
        template <std::size_t trank>
        tensor_size_t size(const tensor_dims_t<trank>& dims)
        {
                static_assert(trank >= 1, "invalid number of tensor dimensions");
                return detail::product(dims, 0);
        }

        ///
        /// \brief compare two tensor dimensions.
        ///
        template <std::size_t trank>
        bool operator==(const tensor_dims_t<trank>& dims1, const tensor_dims_t<trank>& dims2)
        {
                return std::operator==(dims1, dims2);
        }

        template <std::size_t trank>
        bool operator!=(const tensor_dims_t<trank>& dims1, const tensor_dims_t<trank>& dims2)
        {
                return std::operator!=(dims1, dims2);
        }

        ///
        /// \brief stream tensor dimensions.
        ///
        template <std::size_t trank>
        std::ostream& operator<<(std::ostream& os, const tensor_dims_t<trank>& dims)
        {
                for (std::size_t d = 0; d < dims.size(); ++ d)
                {
                        os << dims[d] << (d + 1 == dims.size() ? "" : "x");
                }
                return os;
        }

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
