#pragma once

#include <array>
#include <numeric>

namespace tensor
{
        ///
        /// \brief tensor index
        ///
        template
        <
                typename tindex,
                int tdimensions
        >
        class tensor_index_t
        {
        public:

                static_assert(tdimensions > 0, "tensors cannot have negative number of dimensions");

                ///
                /// \brief constructor
                ///
                template <typename... tindices>
                tensor_index_t(const tindices... sizes) :
                        m_dims({{sizes...}}),
                        m_size(std::accumulate(m_dims.begin(), m_dims.end(), tindex(1), std::multiplies<tindex>()))
                {
                        static_assert(sizeof...(sizes) == tdimensions, "wrong number of tensor dimensions");
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
                tindex operator()(const tindices... indices) const
                {
                        static_assert(sizeof...(indices) == tdimensions, "missing dimensions when indexing tensor");
                        return get_index(indices..., tdimensions - 1);
                }

                ///
                /// \brief retrieve the size of a given dimension
                ///
                tindex size(const int idim) const
                {
                        assert(idim >= 0 && idim < tdimensions);
                        return m_dims[static_cast<std::size_t>(idim)];
                }

                ///
                /// \brief retrieve the tensor size
                ///
                tindex size() const
                {
                        return m_size;
                }

                ///
                /// \brief retrieve the number of dimensions
                ///
                constexpr int dimensionality() const
                {
                        return tdimensions;
                }

        private:

                tindex get_index(const tindex index, const int idim) const
                {
                        assert(idim == 0);
                        assert(index >= 0 && index < size(idim));
                        return index;
                }

                template <typename... tindices>
                tindex get_index(const tindices... indices, const tindex index, const int idim) const
                {
                        assert(index >= 0 && index < size(idim));
                        return index + size(idim) * get_index(indices..., idim - 1);
                }

        private:

                // attributes
                std::array<tindex, tdimensions> m_dims;
                tindex                          m_size;
        };
}
