#include "tensor3d.h"
#include "core/random.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        tensor3d_t::tensor3d_t(size_t dim1, size_t rows, size_t cols)
        {
                resize(dim1, rows, cols);
        }

        //-------------------------------------------------------------------------------------------------

        size_t tensor3d_t::resize(size_t dim1, size_t rows, size_t cols)
        {
                m_dim1 = dim1;
                m_rows = rows;
                m_cols = cols;

                m_data.resize(dim1);
                for (matrix_t& mat : m_data)
                {
                        mat.resize(rows, cols);
                        mat.setZero();
                }

                return size();
        }

        //-------------------------------------------------------------------------------------------------

        void tensor3d_t::zero()
        {
                for (matrix_t& mat : m_data)
                {
                        mat.setZero();
                }
        }

        //-------------------------------------------------------------------------------------------------

        void tensor3d_t::random(scalar_t min, scalar_t max)
        {
                random_t<scalar_t> rgen(min, max);

                for (matrix_t& mat : m_data)
                {
                        rgen(mat.data(), mat.data() + mat.size());
                }
        }

        //-------------------------------------------------------------------------------------------------

        void tensor3d_t::operator +=(const tensor3d_t& other)
        {
                assert(n_dim1() == other.n_dim1());
                assert(n_rows() == other.n_rows());
                assert(n_cols() == other.n_cols());

                for (size_t d1 = 0; d1 < n_dim1(); d1 ++)
                {
                        this->operator()(d1).noalias() += other.operator()(d1);
                }
        }

        //-------------------------------------------------------------------------------------------------
}

