#pragma once

#include <array>
#include <cassert>
#include <ostream>
#include <eigen3/Eigen/Core>

namespace nano
{
        using tensor_size_t = Eigen::Index;

        ///
        /// \brief dimensions of a multi-dimensional tensor.
        ///
        template <std::size_t tdims>
        using tensor_dims_t = std::array<tensor_size_t, tdims>;

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
                template <std::size_t tdims>
                tensor_size_t product(const tensor_dims_t<tdims>& dims, const std::size_t idim)
                {
                        return (idim >= tdims) ? 1 : dims[idim] * product(dims, idim + 1);
                }

                template <std::size_t tdims>
                tensor_size_t get_index(const tensor_dims_t<tdims>&, const std::size_t, const tensor_size_t index)
                {
                        return index;
                }

                template <std::size_t tdims, typename... tindices>
                tensor_size_t get_index(const tensor_dims_t<tdims>& dims, const std::size_t idim,
                        const tensor_size_t index, const tindices... indices)
                {
                        assert(index >= 0 && index < dims[idim]);
                        return index * product(dims, idim + 1) + get_index(dims, idim + 1, indices...);
                }
        }

        ///
        /// \brief index a multi-dimensional tensor.
        ///
        template <std::size_t tdims, typename... tindices>
        tensor_size_t index(const tensor_dims_t<tdims>& dims, const tindices... indices)
        {
                static_assert(tdims >= 1, "invalid number of tensor dimensions");
                static_assert(sizeof...(indices) == tdims, "invalid number of tensor indices");
                return detail::get_index(dims, 0, indices...);
        }

        ///
        /// \brief size of a multi-dimensional tensor (#elements).
        ///
        template <std::size_t tdims>
        tensor_size_t size(const tensor_dims_t<tdims>& dims)
        {
                static_assert(tdims >= 1, "invalid number of tensor dimensions");
                return detail::product(dims, 0);
        }

        ///
        /// \brief compare two tensor dimensions.
        ///
        template <std::size_t tdims>
        bool operator==(const tensor_dims_t<tdims>& dims1, const tensor_dims_t<tdims>& dims2)
        {
                return std::operator==(dims1, dims2);
        }

        template <std::size_t tdims>
        bool operator!=(const tensor_dims_t<tdims>& dims1, const tensor_dims_t<tdims>& dims2)
        {
                return std::operator!=(dims1, dims2);
        }

        ///
        /// \brief stream tensor dimensions.
        ///
        template <std::size_t tdims>
        std::ostream& operator<<(std::ostream& os, const tensor_dims_t<tdims>& dims)
        {
                for (std::size_t d = 0; d < dims.size(); ++ d)
                {
                        os << dims[d] << (d + 1 == dims.size() ? "" : "x");
                }
                return os;
        }

        ///
        /// \brief indexing operations for tensors.
        ///
        template <std::size_t tdimensions>
        struct tensor_indexer_t
        {
                static_assert(tdimensions >= 1, "cannot create tensors with fewer than one dimension");

                using tdims = tensor_dims_t<tdimensions>;
                using Index = tensor_size_t;

                ///
                /// \brief constructor
                ///
                tensor_indexer_t()
                {
                        m_dims.fill(0);
                }

                ///
                /// \brief constructor
                ///
                explicit tensor_indexer_t(const tdims& dims) :
                        m_dims(dims)
                {
                }

                ///
                /// \brief constructor
                ///
                template <typename... tsizes>
                explicit tensor_indexer_t(const tsizes... dims) :
                        m_dims({dims...})
                {
                }

                ///
                /// \brief number of dimensions (aka the rank of the tensor)
                ///
                static auto rank() { return tdimensions; }

                ///
                /// \brief list of dimensions
                ///
                const tdims& dims() const { return m_dims; }

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
                /// \brief compute the linearized index from the list of offsets
                ///
                template <typename... tindices>
                auto offset(const tindices... offsets) const
                {
                        return nano::index(dims(), offsets...);
                }

        protected:

                // attributes
                tdims           m_dims;         ///< dimensions
        };
}
