#ifndef NANOCV_CUDA_MATRIX_H
#define NANOCV_CUDA_MATRIX_H

#include "vector.hpp"

namespace ncv
{
        namespace cuda
        {
                ///
                /// \brief allocated 2D buffer on the device.
                ///
                template
                <
                        typename tscalar
                >
                class matrix_t : public vector_t<tscalar>
                {
                public:

                        ///
                        /// \brief constructor
                        ///
                        matrix_t(int rows, int cols)
                                :       vector_t<tscalar>(rows * cols),
                                        m_rows(rows),
                                        m_cols(cols)
                        {
                        }

                        ///
                        /// \brief disable copying
                        ///
                        matrix_t(const matrix_t&);
                        matrix_t& operator=(const matrix_t&);

                        ///
                        /// \brief access functions
                        ///
                        int rows() const { return m_rows; }
                        int cols() const { return m_cols; }
                        
                        tscalar operator()(int r, int c) const
                        { 
                                return vector_t<tscalar>::operator()(r * m_cols + c);                                 
                        }
                        tscalar& operator()(int r, int c)
                        { 
                                return vector_t<tscalar>::operator()(r * m_cols + c);                                 
                        }

                private:

                        // attributes
                        int             m_rows;
                        int             m_cols;
                };
        }
}

#endif // NANOCV_CUDA_MATRIX_H

