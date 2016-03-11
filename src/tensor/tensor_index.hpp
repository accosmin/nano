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
                        tindex stride = 1;
                        for (int idim = tdimensions - 1; idim >= 0; stride *= size(idim), idim --)
                        {
                                m_strides[static_cast<std::size_t>(idim)] = stride;
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
                        static_assert(sizeof...(indices) + 1 == tdimensions, "missing dimensions when indexing tensor");
                        return get_index<0>(index, indices...);
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

                template <int idim>
                tindex get_index() const
                {
                        return tindex(0);
                }

                template <int idim, typename... tindices>
                tindex get_index(const tindex index, const tindices... indices) const
                {
                        assert(index >= 0 && index < size(idim));
                        if (idim + 1 == tdimensions)
                        {
                                return index;
                        }
                        else
                        {
                                return index * m_strides[idim] + get_index<idim + 1>(indices...);
                        }
                }

        private:

                // attributes
                std::array<tindex, tdimensions> m_dims;
                std::array<tindex, tdimensions> m_strides;
                tindex                          m_size;
        };
}
