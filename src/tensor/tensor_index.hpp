#pragma once

#include <array>
#include <numeric>

namespace tensor
{
        namespace detail
        {
                template <typename tsize>
                tsize dsize()
                {
                        return tsize(1);
                }

                template <typename tsize, typename... tsizes>
                tsize dsize(const tsize dim, const tsizes... dims)
                {
                        return dim * dsize(dims...);
                }
        }

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

                using tindices = std::array<tindex, tdimensions>;

                ///
                /// \brief constructor
                ///
                tensor_index_t()
                {
                        m_sizes.fill(0);
                        m_strides.fill(0);
                }

                ///
                /// \brief constructor
                ///
                template <typename... tindices>
                tensor_index_t(const tindices... sizes) :
                        m_sizes({{sizes...}})
                {
                        static_assert(sizeof...(sizes) == tdimensions, "wrong number of tensor dimensions");
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
                        static_assert(sizeof...(indices) + 1 == tdimensions, "missing dimensions when indexing tensor");
                        return get_index<0>(index, indices...);
                }

                ///
                /// \brief retrieve the size of a given dimension
                ///
                template <int idim>
                tindex size() const
                {
                        static_assert(idim >= 0 && idim < tdimensions, "wrong tensor dimension");
                        return m_sizes[static_cast<std::size_t>(idim)];
                }

                ///
                /// \brief retrieve the tensor size
                ///
                tindex size() const
                {
                        return std::accumulate(m_sizes.begin(), m_sizes.end(), 1, std::multiplies<tindex>());
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
                        assert(index >= 0 && index < size<idim>());
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
                tindices        m_sizes;        ///< size for each dimension
                tindices        m_strides;      ///< stride for each dimension
        };
}
