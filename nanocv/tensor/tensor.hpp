#pragma once

#include "tensor_map.hpp"
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
                        typename tvector = vector_t<tscalar>
                >
                class tensor_t : public tensor_base_t<tvector>
                {
                public:

                        typedef tensor_base_t<tvector>  tbase;
                        typedef typename tbase::tsize   tsize;

                        // Eigen compatible
                        typedef typename tbase::Scalar  Scalar;
                        typedef typename tbase::Index   Index;

                        ///
                        /// \brief constructor
                        ///
                        tensor_t(tsize dims = 0, tsize rows = 0, tsize cols = 0)
                                :       tbase(dims, rows, cols)
                        {
                                resize(dims, rows, cols);
                        }

                        ///
                        /// \brief constructor
                        ///
                        template
                        <
                                typename tscalar_
                        >
                        tensor_t(const tensor_map_t<tscalar_>& other)
                                :       tbase(other.dims(), other.rows(), other.cols())
                        {
                                this->vector() = other.vector().template cast<tscalar>();
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

                        ///
                        /// \brief cast to another tensor type
                        ///
                        template
                        <
                                typename tscalar_
                        >
                        tensor_t<tscalar_> cast() const
                        {
                                tensor_t<tscalar_> copy(this->dims(), this->rows(), this->cols());
                                copy.vector() = this->vector().template cast<tscalar_>();
                                return copy;
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
        }
}
