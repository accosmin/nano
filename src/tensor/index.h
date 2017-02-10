#pragma once

#include <array>
#include <cassert>
#include <ostream>
#include <eigen3/Eigen/Core>

namespace tensor
{
        using index_t = Eigen::Index;

        ///
        /// \brief dimenions of a multi-dimensional tensor.
        ///
        template <std::size_t tdims>
        using dims_t = std::array<index_t, tdims>;

        ///
        /// \brief create a dimension structure for a tensor.
        ///
        template <typename... tsizes>
        auto make_dims(const tsizes... sizes)
        {
                return dims_t<sizeof...(sizes)>({{sizes...}});
        }

        namespace detail
        {
                template <std::size_t tdims>
                index_t get_index(const dims_t<tdims>&, const std::size_t, const index_t index)
                {
                        return index;
                }

                template <std::size_t tdims, typename... tindices>
                index_t get_index(const dims_t<tdims>& dims, const std::size_t idim,
                        const index_t index, const tindices... indices)
                {
                        assert(index >= 0 && index < dims[idim]);
                        assert(idim >= 1);
                        return index + dims[idim] * get_index(dims, idim - 1, indices...);
                }

                template <std::size_t tdims>
                index_t get_size(const dims_t<tdims>& dims, const std::size_t idim)
                {
                        return (idim == tdims) ? 1 : dims[idim] * get_size(dims, idim + 1);
                }
        }

        ///
        /// \brief index a multi-dimensional tensor.
        ///
        template <std::size_t tdims, typename... tindices>
        index_t index(const dims_t<tdims>& dims, const tindices... indices)
        {
                static_assert(tdims >= 1, "invalid number of tensor dimensions");
                static_assert(sizeof...(indices) == tdims, "invalid number of tensor indices");
                return detail::get_index(dims, dims.size() - 1, indices...);
        }

        ///
        /// \brief size of multi-dimensional tensor (#elements).
        ///
        template <std::size_t tdims>
        index_t size(const dims_t<tdims>& dims)
        {
                static_assert(tdims >= 1, "invalid number of tensor dimensions");
                return detail::get_size(dims, 0);
        }

        ///
        /// \brief compare two tensor dimensions.
        ///
        template <std::size_t tdims>
        bool operator==(const dims_t<tdims>& dims1, const dims_t<tdims>& dims2)
        {
                return std::operator==(dims1, dims2);
        }

        template <std::size_t tdims>
        bool operator!=(const dims_t<tdims>& dims1, const dims_t<tdims>& dims2)
        {
                return std::operator!=(dims1, dims2);
        }

        ///
        /// \brief stream tensor dimensions.
        ///
        template <std::size_t tdims>
        std::ostream& operator<<(std::ostream& os, const dims_t<tdims>& dims)
        {
                for (std::size_t d = 0; d < dims.size(); ++ d)
                {
                        os << dims[d] << (d + 1 == dims.size() ? "" : "x");
                }
                return os;
        }
}
