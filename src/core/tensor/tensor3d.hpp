#ifndef NANOCV_TENSOR_TENSOR3D_HPP
#define NANOCV_TENSOR_TENSOR3D_HPP

#include "core/tensor/storage.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // 3D tensor:
        //      - 1D/2D collection of fixed size matrices
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace tensor
        {                
                template
                <
                        typename tscalar,
                        typename tsize
                >
                class tensor3d_t : public tensor::storage_t<tscalar, tsize>
                {
                public:

                        typedef tensor::storage_t<tscalar, tsize>       base_t;
                        typedef typename base_t::matrix_t               matrix_t;

                        // constructor
                        tensor3d_t(tsize dim1 = 0, tsize rows = 0, tsize cols = 0)
                        {
                                resize(dim1, rows, cols);
                        }

                        // resize to new dimensions
                        tsize resize(tsize dim1, tsize rows, tsize cols)
                        {
                                m_dim1 = dim1;
                                return base_t::allocate(dim1, rows, cols);
                        }

                        // access functions
                        tsize n_dim1() const { return m_dim1; }

                        const matrix_t& operator()(tsize d1) const
                        {
                                return base_t::get(d1);
                        }
                        matrix_t& operator()(tsize d1)
                        {
                                return base_t::get(d1);
                        }

                        const tscalar& operator()(tsize d1, size_t r, size_t c) const
                        {
                                return base_t::get(d1)(r, c);
                        }
                        tscalar& operator()(tsize d1, size_t r, size_t c)
                        {
                                return base_t::get(d1)(r, c);
                        }

                private:

                        // serialize
                        friend class boost::serialization::access;
                        template
                        <
                                class tarchive
                        >
                        void serialize(tarchive & ar, const unsigned int version)
                        {
                                ar & boost::serialization::base_object<base_t>(*this);
                                ar & m_dim1;
                        }

                private:

                        // attributes
                        tsize           m_dim1; // #dimension 1
                };
        }
}

// serialize 3D tensor
namespace boost
{
        namespace serialization
        {
                template
                <
                        class tarchive,
                        class tscalar,
                        class tsize
                >
                void serialize(tarchive& ar, ncv::tensor::tensor3d_t<tscalar, tsize>& t3, const unsigned int version)
                {
                        ar & t3;
                }
        }
}

#endif // NANOCV_TENSOR_TENSOR3D_HPP
