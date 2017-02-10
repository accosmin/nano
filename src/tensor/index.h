#pragma once

#include <array>
#include <cassert>
#include <ostream>

namespace tensor
{
        ///
        /// \brief dimenions of a multi-dimensional tensor.
        ///
        template <typename tindex, std::size_t tdims>
        using dims_t = std::array<tindex, tdims>;

        namespace detail
        {
                template <typename tindex, std::size_t tdims>
                tindex get_index(const dims_t<tindex, tdims>&, const std::size_t)
                {
                        return 1;
                }

                template <typename tindex, std::size_t tdims, typename... tindices>
                tindex get_index(const dims_t<tindex, tdims>& dims, const std::size_t idim,
                        const tindices... indices, const tindex index)
                {
                        assert(index >= 0 && index < dims[idim]);
                        assert(idim >= 1);
                        return index + dims[idim] * get_index(dims, idim - 1, indices...);
                }

                template <typename tindex, std::size_t tdims>
                tindex get_size(const dims_t<tindex, tdims>& dims, const std::size_t idim)
                {
                        return (idim == tdims) ? 1 : dims[idim] * get_size(dims, idim + 1);
                }
        }

        ///
        /// \brief index a multi-dimensional tensor.
        ///
        template <typename tindex, std::size_t tdims, typename... tindices>
        tindex index(const dims_t<tindex, tdims>& dims, const tindices... indices)
        {
                static_assert(tdims > 1, "invalid number of tensor dimensions");
                static_assert(sizeof...(indices) == tdims, "invalid number of tensor indices");
                return detail::get_index(dims, dims.size() - 1, indices...);
        }

        ///
        /// \brief size of multi-dimensional tensor (#elements).
        ///
        template <typename tindex, std::size_t tdims>
        tindex size(const dims_t<tindex, tdims>& dims)
        {
                static_assert(tdims > 1, "invalid number of tensor dimensions");
                return detail::get_size(dims, 0);
        }

        ///
        /// \brief compare two tensor dimensions.
        ///
        template <typename tindex, std::size_t tdims>
        bool operator==(const dims_t<tindex, tdims>& dims1, const dims_t<tindex, tdims>& dims2)
        {
                return std::operator==(dims1, dims2);
        }

        template <typename tindex, std::size_t tdims>
        bool operator!=(const dims_t<tindex, tdims>& dims1, const dims_t<tindex, tdims>& dims2)
        {
                return std::operator!=(dims1, dims2);
        }

        ///
        /// \brief stream tensor dimensions.
        ///
        template <typename tindex, std::size_t tdims>
        std::ostream& operator<<(std::ostream& os, const dims_t<tindex, tdims>& dims)
        {
                for (std::size_t d = 0; d < dims.size(); ++ d)
                {
                        os << dims[d] << (d + 1 == dims.size() ? "" : "x");
                }
                return os;
        }
}
