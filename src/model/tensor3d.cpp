#include "tensor3d.h"

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

        vector_t tensor3d_t::to_vector() const
        {
                assert(1 == n_rows());
                assert(1 == n_cols());

                vector_t result(n_dim1());

                for (size_t d1 = 0; d1 < n_dim1(); d1 ++)
                {
                        result(d1) = data(d1)(0, 0);
                }

                return result;
        }

        //-------------------------------------------------------------------------------------------------

        void tensor3d_t::from_vector(const vector_t& vec)
        {
                assert(1 == n_rows());
                assert(1 == n_cols());
                assert(v.size() == n_dim1());

                for (size_t d1 = 0; d1 < n_dim1(); d1 ++)
                {
                        data(d1)(0, 0) = vec(d1);
                }
        }

        //-------------------------------------------------------------------------------------------------

        void tensor3d_t::serialize(serializer_t& s) const
        {
                s << m_data;
        }

        //-------------------------------------------------------------------------------------------------

        void tensor3d_t::deserialize(deserializer_t& s)
        {
                s >> m_data;
        }

        //-------------------------------------------------------------------------------------------------
}

