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
                static constexpr auto rank() { return trank; }

                ///
                /// \brief list of dimensions
                ///
                const auto& dims() const { return m_dims; }

                ///
                /// \brief gather the missing dimensions in a multi-dimensional tensor
                ///     (assuming the last dimensions that are ignored are zero).
                ///
                template <typename... tindices>
                auto dims0(const tindices... indices) const
                {
                        static_assert(sizeof...(indices) <= trank, "invalid number of tensor dimensions");
                        return nano::dims0(dims(), indices...);
                }

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
                auto offset(const tindices... indices) const
                {
                        static_assert(sizeof...(indices) == trank, "invalid number of tensor dimensions");
                        return nano::index(dims(), indices...);
                }

                ///
                /// \brief compute the linearized index from the list of offsets
                ///     (assuming the last dimensions that are ignored are zero)
                ///
                template <typename... tindices>
                auto offset0(const tindices... indices) const
                {
                        static_assert(sizeof...(indices) <= trank, "invalid number of tensor dimensions");
                        return nano::index0(dims(), indices...);
                }

        protected:

                template <typename tarray>
                static void zero(tarray&& array)
                {
                        array.setZero();
                }

                template <typename tarray, typename tscalar>
                static void constant(tarray&& array, const tscalar value)
                {
                        array.setConstant(value);
                }

                template <typename tarray, typename tscalar>
                static void random(tarray&& array, const tscalar min, const tscalar max)
                {
                        assert(min < max);
                        array.setRandom(); // [-1, +1]
                        array = (array + 1) * (max - min) / 2 + min;
                }

                template <typename tdata, typename... tindices>
                auto data(tdata ptr, const tindices... indices) const
                {
                        static_assert(sizeof...(indices) <= trank, "invalid number of tensor dimensions");
                        return ptr + offset0(indices...);
                }

                template <typename tdata, typename... tindices>
                auto vector(tdata ptr, const tindices... indices) const
                {
                        static_assert(sizeof...(indices) < trank, "invalid number of tensor dimensions");
                        return map_vector(data(ptr, indices...), nano::size(nano::dims0(dims(), indices...)));
                }

                template <typename tdata, typename... tindices>
                auto array(tdata ptr, const tindices... indices) const
                {
                        static_assert(sizeof...(indices) < trank, "invalid number of tensor dimensions");
                        return vector(ptr, indices...).array();
                }

                template <typename tdata, typename... tindices>
                auto matrix(tdata ptr, const tindices... indices) const
                {
                        static_assert(sizeof...(indices) == trank - 2, "invalid number of tensor dimensions");
                        return map_matrix(data(ptr, indices..., 0, 0), rows(), cols());
                }

                // attributes
                tdims           m_dims;         ///< dimensions
        };
}
