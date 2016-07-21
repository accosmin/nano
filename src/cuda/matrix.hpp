#pragma once

#include "vector.hpp"

namespace nano
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
                        matrix_t(int rows = 0, int cols = 0)
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
                        /// \brief resize to new dimensions
                        ///
                        bool resize(int rows, int cols)
                        {
                                if (vector_t<tscalar>::resize(rows * cols))
                                {
                                        m_rows = rows;
                                        m_cols = cols;
                                        return true;
                                }
                                else
                                {
                                        return false;
                                }
                        }

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
                        int                     m_rows;
                        int                     m_cols;
                };

                typedef matrix_t<int>           imatrix_t;
                typedef matrix_t<float>         fmatrix_t;
                typedef matrix_t<double>        dmatrix_t;
        }
}

