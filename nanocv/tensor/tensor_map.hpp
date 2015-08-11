#pragma once

#include "tensor_base.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief 3D tensor mapping an array as ::dims() 2D planes of size ::rows() x ::cols()
                ///
                template
                <
                        typename tscalar,
                        typename tvector = vector_t<tscalar>,
                        typename tmap = Eigen::Map<tvector>
                >
                class tensor_map_t : public tensor_base_t<tmap>
                {
                public:

                        typedef tensor_base_t<tmap>     tbase;
                        typedef typename tbase::tsize   tsize;

                        // Eigen compatible
                        typedef typename tbase::Scalar  Scalar;
                        typedef typename tbase::Index   Index;

                        ///
                        /// \brief constructor
                        ///
                        tensor_map_t(tscalar* data, tsize dims, tsize rows, tsize cols)
                                :       tbase(dims, rows, cols, tensor::map_vector(data, dims * rows * cols))
                        {
                        }
                };

                ///
                /// \brief map non-constant data to tensors
                ///
                template
                <
                        typename tvalue_,
                        typename tsize,
                        typename tvalue = typename std::remove_const<tvalue_>::type,
                        typename tresult = tensor_map_t<tvalue>
                >
                tresult map_tensor(tvalue_* data, tsize dims, tsize rows, tsize cols)
                {
                        return tresult(data, dims, rows, cols);
                }

                ///
                /// \brief map constant data to tensors
                ///
                template
                <
                        typename tvalue_,
                        typename tsize,
                        typename tvalue = typename std::remove_const<tvalue_>::type,
                        typename tresult = tensor_map_t<tvalue>
                >
                tresult map_tensor(const tvalue_* data, tsize dims, tsize rows, tsize cols)
                {
                        return tresult(data, dims, rows, cols);
                }
        }
}
