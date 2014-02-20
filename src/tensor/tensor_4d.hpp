#ifndef NANOCV_TENSOR_4D_HPP
#define NANOCV_TENSOR_4D_HPP

#include "tensor_base.hpp"
#include <boost/serialization/base_object.hpp>

namespace ncv
{
        namespace tensor
        {
                template
                <
                        typename tscalar,
                        typename tsize
                >
                class tensor_4d_t : public tensor::base_t<tscalar, tsize>
                {
                public:

                        typedef tensor::base_t<tscalar, tsize>          tbase;

                        ///
                        /// \brief constructor
                        ///
                        tensor_4d_t(tsize dim1 = 0, tsize dim2 = 0, tsize rows = 0, tsize cols = 0)
                        {
                                resize(dim1, dim2, rows, cols);
                        }

                        ///
                        /// \brief resize to new dimensions
                        ///
                        tsize resize(tsize dim1, tsize dim2, tsize rows, tsize cols)
                        {
                                m_dim1 = dim1;
                                m_dim2 = dim2;
                                return tbase::allocate(dim1 * dim2, rows, cols);
                        }

                        ///
                        /// \brief 4D dimensions
                        ///
                        tsize dim1() const { return m_dim1; }
                        tsize dim2() const { return m_dim2; }

                        ///
                        /// \brief access a plane as raw data
                        ///
                        const tscalar* plane(tsize i1, tsize i2) const { return tbase::plane_data(plane_index(i1, i2)); }
                        tscalar* plane(tsize i1, tsize i2) { return tbase::plane_data(plane_index(i1, i2)); }

                private:

                        ///
                        /// \brief compute the matrix/plane index
                        ///
                        tsize plane_index(tsize i1, tsize i2) const
                        {
                                return i1 * dim2() + i2;
                        }

                        ///
                        /// \brief serialize
                        ///
                        friend class boost::serialization::access;
                        template
                        <
                                class tarchive
                        >
                        void serialize(tarchive & ar, const unsigned int)
                        {
                                ar & boost::serialization::base_object<tbase>(*this);
                                ar & m_dim1;
                                ar & m_dim2;
                        }

                private:

                        // attributes
                        tsize           m_dim1;         ///< #dimension 1
                        tsize           m_dim2;         ///< #dimension 2
                };
        }
}

#endif // NANOCV_TENSOR_TENSOR4D_HPP
