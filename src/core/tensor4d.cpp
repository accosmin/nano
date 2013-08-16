#include "tensor4d.h"
#include "core/random.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        tensor4d_t::tensor4d_t(size_t dim1, size_t dim2, size_t rows, size_t cols)
        {
                resize(dim1, dim2, rows, cols);
        }

        //-------------------------------------------------------------------------------------------------

        size_t tensor4d_t::resize(size_t dim1, size_t dim2, size_t rows, size_t cols)
        {
                m_dim1 = dim1;
                m_dim2 = dim2;
                m_rows = rows;
                m_cols = cols;

                m_data.resize(dim1 * dim2);
                for (matrix_t& mat : m_data)
                {
                        mat.resize(rows, cols);
                        mat.setZero();
                }

                return size();
        }

        //-------------------------------------------------------------------------------------------------

        void tensor4d_t::zero()
        {
                constant(0.0);
        }

        //-------------------------------------------------------------------------------------------------

        void tensor4d_t::constant(scalar_t value)
        {
                for (matrix_t& mat : m_data)
                {
                        mat.setConstant(value);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void tensor4d_t::random(scalar_t min, scalar_t max)
        {
                random_t<scalar_t> rgen(min, max);

                for (matrix_t& mat : m_data)
                {
                        rgen(mat.data(), mat.data() + mat.size());
                }
        }

        //-------------------------------------------------------------------------------------------------

        void tensor4d_t::operator +=(const tensor4d_t& other)
        {
                assert(n_dim1() == other.n_dim1());
                assert(n_dim2() == other.n_dim2());
                assert(n_rows() == other.n_rows());
                assert(n_cols() == other.n_cols());

                for (size_t d1 = 0; d1 < n_dim1(); d1 ++)
                {
                        for (size_t d2 = 0; d2 < n_dim2(); d2 ++)
                        {
                                this->operator()(d1, d2).noalias() += other.operator()(d1, d2);
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        serializer_t& operator<<(serializer_t& s, const tensor4d_t& tensor)
        {
                return s << tensor.m_data;
        }

        //-------------------------------------------------------------------------------------------------

        deserializer_t& operator>>(deserializer_t& s, tensor4d_t& tensor)
        {
                return s >> tensor.m_data;
        }

        //-------------------------------------------------------------------------------------------------
}

