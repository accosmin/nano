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

                m_data.resize(dim1);
                for (matrices_t& mats : m_data)
                {
                        mats.resize(dim2);

                        for (matrix_t& mat : mats)
                        {
                                mat.resize(rows, cols);
                                mat.setZero();
                        }
                }

                return size();
        }

        //-------------------------------------------------------------------------------------------------

        void tensor4d_t::zero()
        {
                for (matrices_t& mats : m_data)
                {
                        for (matrix_t& mat : mats)
                        {
                                mat.setZero();
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        void tensor4d_t::random(scalar_t min, scalar_t max)
        {
                random_t<scalar_t> rgen(min, max);

                for (matrices_t& mats : m_data)
                {
                        for (matrix_t& mat : mats)
                        {
                                rgen(mat.data(), mat.data() + mat.size());
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        tensor3d_t tensor4d_t::forward(const tensor3d_t& input) const
        {
                assert(dim1() == input.dim1());
                assert(rows()  < input.rows());
                assert(cols()  < input.cols());

                const size_t orows = input.rows() - rows();
                const size_t ocols = input.cols() - cols();

                tensor3d_t result(dim2, orows, ocols);

                for (size_t d = 0; d < dims(); d ++)
                {
                        const matrix_t& in = input[d1];
                        const matrix_t& co = m_data[d1];

                        for (size_t r = 0; r < orows; r ++)
                        {
                                for (size_t c = 0; c < ocols; c ++)
                                {
                                        result(r, c) +=
                                                input.block(r, c, rows(), cols()).cwiseProduct(convo).sum();
                                }
                        }
                }

//                return result;

                return tensor3d_t();
        }

        //-------------------------------------------------------------------------------------------------

        tensor3d_t tensor4d_t::backward(const tensor3d_t& gradient) const
        {

                return tensor3d_t();
        }

        //-------------------------------------------------------------------------------------------------

        tensor4d_t tensor4d_t::gradient(const tensor3d_t& input, const tensor3d_t& gradient) const
        {

                return tensor4d_t();
        }

        //-------------------------------------------------------------------------------------------------

        void tensor4d_t::serialize(serializer_t& s) const
        {
                for (size_t d1 = 0; d1 < dim1(); d1 ++)
                {
                        s << m_data[d1];
                }
        }

        //-------------------------------------------------------------------------------------------------

        void tensor4d_t::deserialize(deserializer_t& s)
        {
                for (size_t d1 = 0; d1 < dim1(); d1 ++)
                {
                        s >> m_data[d1];
                }
        }

        //-------------------------------------------------------------------------------------------------

        void tensor4d_t::operator +=(const tensor4d_t& other)
        {
                assert(dim1() == other.dim1());
                assert(dim2() == other.dim2());
                assert(rows() == other.rows());
                assert(cols() == other.cols());

                for (size_t d1 = 0; d1 < dim1(); d1 ++)
                {
                        for (size_t d2 = 0; d2 < dim2(); d2 ++)
                        {
                                m_data[d1][d2].noalias() += other(d1, d2);
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------
}

