#pragma once

#include "tensor_base.hpp"
#include "nanocv/arch.h"
#include <boost/serialization/access.hpp>

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief 3D tensor stored as ::dims() 2D planes of size ::rows() x ::cols()
                ///
                template
                <
                        typename tscalar,
                        typename tsize,
                        typename tindex = tsize,
                        typename tvector = typename vector_types_t<tscalar>::tvector,
                        typename tbase = tensor_base_t<tscalar, tsize, tvector>
                >
                class tensor_t : public tbase
                {
                public:

                        // Eigen compatible
                        typedef typename tbase::Scalar Scalar;

                        ///
                        /// \brief constructor
                        ///
                        tensor_t(tsize dims = 0, tsize rows = 0, tsize cols = 0)
                                :       tbase(dims, rows, cols)
                        {
                                resize(dims, rows, cols);
                        }

                        ///
                        /// \brief resize to new dimensions
                        ///
                        tsize resize(tsize dims, tsize rows, tsize cols)
                        {
                                this->m_dims = dims;
                                this->m_rows = rows;
                                this->m_cols = cols;
                                this->m_data.resize(dims * rows * cols);

                                return this->size();
                        }

                private:

                        ///
                        /// \brief serialize
                        ///
                        friend class boost::serialization::access;
                        template
                        <
                                class tarchive
                        >
                        void serialize(tarchive & ar, const unsigned int version)
                        {
                                NANOCV_UNUSED1(version);

                                ar & this->m_dims;
                                ar & this->m_rows;
                                ar & this->m_cols;
                                ar & this->m_data;
                        }
                };

                ///
                /// \brief 3D tensor mapping an array as ::dims() 2D planes of size ::rows() x ::cols()
                ///
                template
                <
                        typename tscalar,
                        typename tsize,
                        typename tindex = tsize,
                        typename tvector = typename vector_types_t<tscalar>::tvector,
                        typename tmap = Eigen::Map<tvector>,
                        typename tbase = tensor_base_t<tscalar, tsize, tmap>
                >
                class tensor_map_t : public tbase
                {
                public:

                        // Eigen compatible
                        typedef typename tbase::Scalar Scalar;

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
                        typename tresult = tensor_map_t<tvalue, tsize>
                >
                tresult map_tensor(tvalue_* data, tsize dims, tsize rows, tsize cols)
                {
                        return tresult(data, dims, rows, cols);
                }

                ///
                /// \brief map non-constant data to tensors (using the dimensions of an equivalent tensor)
                ///
                template
                <
                        typename tvalue_,
                        typename ttensor
                >
                decltype(auto) map_tensor(tvalue_* data, const ttensor& tensor)
                {
                        return map_tensor(data, tensor.dims(), tensor.rows(), tensor.cols());
                }

                ///
                /// \brief map constant data to tensors
                ///
                template
                <
                        typename tvalue_,
                        typename tsize,
                        typename tvalue = typename std::remove_const<tvalue_>::type,
                        typename tresult = tensor_map_t<tvalue, tsize>
                >
                tresult map_tensor(const tvalue_* data, tsize dims, tsize rows, tsize cols)
                {
                        return tresult(data, dims, rows, cols);
                }

                ///
                /// \brief map constant data to tensors (using the dimensions of an equivalent tensor)
                ///
                template
                <
                        typename tvalue_,
                        typename ttensor
                >
                decltype(auto) map_tensor(const tvalue_* data, const ttensor& tensor)
                {
                        return map_tensor(data, tensor.dims(), tensor.rows(), tensor.cols());
                }
        }
}
