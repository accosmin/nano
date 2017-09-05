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
                template <std::size_t idim, std::size_t trank>
                struct product_t
                {
                        static tensor_size_t get(const tensor_dims_t<trank>& dims)
                        {
                                return std::get<idim>(dims) * product_t<idim + 1, trank>::get(dims);
                        }
                };

                template <std::size_t trank>
                struct product_t<trank, trank>
                {
                        static tensor_size_t get(const tensor_dims_t<trank>&)
                        {
                                return 1;
                        }
                };

                template <std::size_t idim, std::size_t trank>
                tensor_size_t product(const tensor_dims_t<trank>& dims)
                {
                        return product_t<idim, trank>::get(dims);
                }

                template <std::size_t idim, std::size_t trank>
                tensor_size_t get_index(const tensor_dims_t<trank>&, const tensor_size_t index)
                {
                        return index;
                }

                template <std::size_t idim, std::size_t trank, typename... tindices>
                tensor_size_t get_index(const tensor_dims_t<trank>& dims,
                        const tensor_size_t index, const tindices... indices)
                {
                        assert(index >= 0 && index < std::get<idim>(dims));
                        return index * product<idim + 1>(dims) + get_index<idim + 1>(dims, indices...);
                }

                template <std::size_t idim, std::size_t trank>
                tensor_size_t get_index0(const tensor_dims_t<trank>&)
                {
                        return 0;
                }

                template <std::size_t idim, std::size_t trank, typename... tindices>
                tensor_size_t get_index0(const tensor_dims_t<trank>& dims,
                        const tensor_size_t index, const tindices... indices)
                {
                        assert(index >= 0 && index < std::get<idim>(dims));
                        return index * product<idim + 1>(dims) + get_index0<idim + 1>(dims, indices...);
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
                return detail::get_index<0>(dims, indices...);
        }

        ///
        /// \brief index a multi-dimensional tensor (assuming the last dimensions that are ignored are zero).
        ///
        template <std::size_t trank, typename... tindices>
        tensor_size_t index0(const tensor_dims_t<trank>& dims, const tindices... indices)
        {
                static_assert(trank >= 1, "invalid number of tensor dimensions");
                static_assert(sizeof...(indices) <= trank, "invalid number of tensor indices");
                return detail::get_index0<0>(dims, indices...);
        }
        ///
        /// \brief index a multi-dimensional tensor (assuming the last dimensions that are ignored are zero).
        /// NB: gather the missing dimensions in dimx.
        ///
        template <std::size_t trank, typename... tindices, std::size_t trankx = trank - sizeof...(tindices)>
        tensor_size_t index0x(const tensor_dims_t<trank>& dims, tensor_dims_t<trankx>& dimx, const tindices... indices)
        ;/*{
                static_assert(trank >= 1, "invalid number of tensor dimensions");
                static_assert(sizeof...(indices) < trank, "invalid number of tensor indices");
                return detail::get_index0x(dims, dimx, 0, indices...);
        }*/

        ///
        /// \brief size of a multi-dimensional tensor (#elements).
        ///
        template <std::size_t trank>
        tensor_size_t size(const tensor_dims_t<trank>& dims)
        {
                static_assert(trank >= 1, "invalid number of tensor dimensions");
                return detail::product<0>(dims);
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
}
