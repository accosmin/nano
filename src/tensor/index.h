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
        }

        ///
        /// \brief index a multi-dimensional tensor.
        ///
        template <typename tindex, std::size_t tdims, typename... tindices>
        tindex get_index(const dims_t<tindex, tdims>& dims, const tindices... indices)
        {
                static_assert(tdims > 1, "invalid number of tensor dimensions");
                static_assert(sizeof...(indices) == tdims, "invalid number of tensor indices");
                return detail::get_index(dims, dims.size() - 1, indices...);
        }

        ///
        /// \brief size of multi-dimensional tensor (#elements).
        ///
        template <typename tindex, std::size_t tdims, typename... tindices>
        tindex size(const dims_t<tindex, tdims>& dims, const tindices... indices)
        {
                static_assert(tdims > 1, "invalid number of tensor dimensions");
                static_assert(sizeof...(indices) == tdims, "invalid number of tensor indices");
                return detail::get_index(dims, dims.size() - 1, indices...);
        }

        namespace detail
        {
                template <typename tvalue, std::size_t tsize>
                constexpr tvalue product_array(const std::array<tvalue, tsize>& a, const std::size_t i = 0)
                {
                        return (i < tsize)? a[i] * product_array(a, i + 1) : tvalue(1);
                }

                template <typename tsize>
                tsize product_variadic()
                {
                        return tsize(1);
                }

                template <typename tsize, typename... tsizes>
                tsize product_variadic(const tsize dim, const tsizes... dims)
                {
                        return dim * product_variadic<tsize>(dims...);
                }
        }

        ///
        /// \brief index a multi-dimensional tensor.
        ///
        template <typename tindex, std::size_t tdims>
        struct index_t
        {
        public:

                static_assert(tdims > 0, "tensors cannot have negative number of dimensions");

                using tindices = std::array<tindex, tdims>;

                ///
                /// \brief constructor
                ///
                index_t()
                {
                        m_sizes.fill(0);
                        m_strides.fill(0);
                }

                ///
                /// \brief constructor
                ///
                template <typename... tindices>
                explicit index_t(const tindices... sizes) :
                        m_sizes({{sizes...}})
                {
                        static_assert(sizeof...(sizes) == tdims, "wrong number of tensor dimensions");
                        tindex stride = 1;
                        for (auto itstride = m_strides.rbegin(), itsize = m_sizes.rbegin();
                                itstride != m_strides.rend(); ++ itstride, ++ itsize)
                        {
                                *itstride = stride;
                                stride *= *itsize;
                        }
                }

                ///
                /// \brief retrieve the linearized index [0, size)
                ///
                tindex operator()(const tindex index) const
                {
                        assert(index >= 0 && index < size());
                        return index;
                }

                ///
                /// \brief retrieve the linearized index [0, size) from the given dimensional indices
                ///
                template <typename... tindices>
                tindex operator()(const tindex index, const tindices... indices) const
                {
                        static_assert(sizeof...(indices) + 1 == tdims, "missing dimensions when indexing tensor");
                        return get_index<0>(index, indices...);
                }

                ///
                /// \brief retrieve the size of a given dimension
                ///
                template <std::size_t idim>
                tindex size() const
                {
                        static_assert(idim < tdims, "wrong tensor dimension");
                        return m_sizes[idim];
                }

                ///
                /// \brief retrieve the tensor size
                ///
                tindex size() const
                {
                        return detail::product_array(m_sizes);
                }

                ///
                /// \brief retrieve the number of dimensions
                ///
                static constexpr std::size_t dimensionality()
                {
                        return tdims;
                }

                ///
                /// \brief retrieve the dimensions
                ///
                const auto& dims() const
                {
                        return m_sizes;
                }

        private:

                template <std::size_t idim>
                tindex get_index() const
                {
                        return tindex(0);
                }

                template <std::size_t idim, typename... tindices>
                tindex get_index(const tindex index, const tindices... indices) const
                {
                        assert(index >= 0 && index < size<idim>());
                        return index * m_strides[idim] + get_index<idim + 1>(indices...);
                }

        private:

                // attributes
                tindices        m_sizes;        ///< size for each dimension
                tindices        m_strides;      ///< stride for each dimension
        };

        ///
        /// \brief compare two tensor dimensions.
        ///
        template <typename tindex, std::size_t tdims>
        bool operator==(const index_t<tindex, tdims>& ti1, const index_t<tindex, tdims>& ti2)
        {
                return std::operator==(ti1.dims(), ti2.dims());
        }

        template <typename tindex, std::size_t tdims>
        bool operator!=(const index_t<tindex, tdims>& ti1, const index_t<tindex, tdims>& ti2)
        {
                return !operator==(ti1, ti2);
        }

        ///
        /// \brief stream tensor dimensions.
        ///
        template <typename tindex, std::size_t tdims>
        std::ostream& operator<<(std::ostream& os, const index_t<tindex, tdims>& ti)
        {
                const auto& dims = ti.dims();
                for (std::size_t d = 0; d < dims.size(); ++ d)
                {
                        os << dims[d] << (d + 1 == dims.size() ? "" : "x");
                }
                return os;
        }
}
