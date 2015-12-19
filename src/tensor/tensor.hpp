#pragma once

#include "tensor_map.hpp"

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

                using tbase = tensor_base_t<tvector>;
                using tsize = typename tbase::tsize;

                // Eigen compatible
                using Index = typename tbase::Index;
                using Scalar = typename tbase::Scalar;

                ///
                /// \brief constructor
                ///
                explicit tensor_t(tsize dims = 0, tsize rows = 0, tsize cols = 0)
                        :       tbase(dims, rows, cols)
                {
                        resize(dims, rows, cols);
                }

                ///
                /// \brief constructor
                ///
                template
                <
                        typename tmap
                >
                tensor_t(const tensor_map_t<tmap>& other)
                        :       tensor_t(other.dims(), other.rows(), other.cols())
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
        };
}
