#ifndef NANOCV_TENSOR_3D_HPP
#define NANOCV_TENSOR_3D_HPP

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
                class tensor3d_t : public tensor::base_t<tscalar, tsize>
                {
                public:

                        typedef tensor::base_t<tscalar, tsize>          tbase;
                        typedef typename tbase::tmatrix                 tmatrix;

                        ///
                        /// \brief constructor
                        ///
                        tensor3d_t(tsize dim1 = 0, tsize rows = 0, tsize cols = 0)
                        {
                                resize(dim1, rows, cols);
                        }

                        ///
                        /// \brief resize to new dimensions
                        ///
                        tsize resize(tsize dim1, tsize rows, tsize cols)
                        {
                                m_dim1 = dim1;
                                return tbase::allocate(dim1, rows, cols);
                        }

                        ///
                        /// \brief 3D dimensions
                        ///
                        tsize dim1() const { return m_dim1; }

                        ///
                        /// \brief access a matrix/plane as raw data
                        ///
                        const tscalar* matrix_data(tsize i) const { return tbase::matrix_data(i); }
                        tscalar* matrix_data(tsize i) { return tbase::matrix_data(i); }

                        ///
                        /// \brief access a matrix/plane as Eigen matrix wrapper
                        ///
                        tmatrix_map as_matrix(tsize i) { return tbase::tmatrix_map(i); }

                private:

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
                        }

                private:

                        // attributes
                        tsize           m_dim1;         ///< #dimension 1
                };
        }
}

#endif // NANOCV_TENSOR_3D_HPP
