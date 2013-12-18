#ifndef NANOCV_TENSOR_TENSOR4D_HPP
#define NANOCV_TENSOR_TENSOR4D_HPP

#include "storage.hpp"
#include <boost/serialization/base_object.hpp>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // 4D tensor:
        //      - 1D/2D collection of fixed size matrices
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace tensor
        {
                template
                <
                        typename tscalar,
                        typename tsize
                >
                class tensor4d_t : public tensor::storage_t<tscalar, tsize>
                {
                public:

                        typedef tensor::storage_t<tscalar, tsize>       tbase;
                        typedef typename tbase::tmatrix                 tmatrix;

                        // constructor
                        tensor4d_t(tsize dim1 = 0, tsize dim2 = 0, tsize rows = 0, tsize cols = 0)
                        {
                                resize(dim1, dim2, rows, cols);
                        }

                        // resize to new dimensions
                        tsize resize(tsize dim1, tsize dim2, tsize rows, tsize cols)
                        {
                                m_dim1 = dim1;
                                m_dim2 = dim2;
                                return tbase::allocate(dim1 * dim2, rows, cols);
                        }

                        // access functions
                        tsize n_dim1() const { return m_dim1; }
                        tsize n_dim2() const { return m_dim2; }

                        const tmatrix& operator()(size_t d1, size_t d2) const
                        {
                                return tbase::get(d1 * n_dim2() + d2);
                        }
                        tmatrix& operator()(size_t d1, size_t d2)
                        {
                                return tbase::get(d1 * n_dim2() + d2);
                        }

                        const tscalar& operator()(tsize d1, tsize d2, size_t r, size_t c) const
                        {
                                return tbase::get(d1, d2)(r, c);
                        }
                        tscalar& operator()(tsize d1, tsize d2, size_t r, size_t c)
                        {
                                return tbase::get(d1, d2)(r, c);
                        }

                private:

                        // serialize
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
                        tsize           m_dim1; // #dimension 1
                        tsize           m_dim2; // #dimension 2
                };
        }
}

#endif // NANOCV_TENSOR_TENSOR4D_HPP
