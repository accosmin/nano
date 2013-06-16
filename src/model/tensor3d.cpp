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

        scalar_t tensor3d_t::forward(const tensor3d_t& input) const
        {
                assert(dim1() == inputs.dim1());
                assert(rows() == inputs.rows());
                assert(cols() == inputs.cols());

                scalar_t result = 0.0;
                for (size_t d1 = 0; d1 < dim1(); d1 ++)
                {
                        const matrix_t& in = input[d1];
                        const matrix_t& co = m_data[d1];

                        result += in.cwiseProduct(co).sum();
                }

                return result;
        }

        //-------------------------------------------------------------------------------------------------

        tensor3d_t tensor3d_t::backward(scalar_t gradient) const
        {
                tensor3d_t result(dim1(), rows(), cols());
                for (size_t d1 = 0; d1 < dim1(); d1 ++)
                {
                        result.m_data[d1] = m_data[d1] * gradient;
                }

                return result;
        }

        //-------------------------------------------------------------------------------------------------

        tensor3d_t tensor3d_t::gradient(const tensor3d_t& input, scalar_t gradient) const
        {
                assert(dim1() == inputs.dim1());
                assert(rows() == inputs.rows());
                assert(cols() == inputs.cols());

                tensor3d_t result(dim1(), rows(), cols());
                for (size_t d1 = 0; d1 < dim1(); d1 ++)
                {
                        result.m_data[d1] = input[d1] * gradient;
                }

                return result;
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

        void tensor3d_t::operator +=(const tensor3d_t& other)
        {
                assert(dim1() == other.dim1());
                assert(rows() == other.rows());
                assert(cols() == other.cols());

                for (size_t d1 = 0; d1 < dim1(); d1 ++)
                {
                        m_data[d1].noalias() += other[d1];
                }
        }

        //-------------------------------------------------------------------------------------------------
}

