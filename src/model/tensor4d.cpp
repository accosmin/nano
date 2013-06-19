#include "tensor4d.h"
#include "core/random.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        static void forward(const matrix_t& in, const matrix_t& co, matrix_t& out)
        {
                const size_t crows = static_cast<size_t>(co.rows());
                const size_t ccols = static_cast<size_t>(co.cols());

                const size_t orows = static_cast<size_t>(in.rows() - crows);
                const size_t ocols = static_cast<size_t>(in.cols() - ccols);

                for (size_t r = 0; r < orows; r ++)
                {
                        for (size_t c = 0; c < ocols; c ++)
                        {
                                out(r, c) += in.block(r, c, crows, ccols).cwiseProduct(co).sum();
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        static void gradient(const matrix_t& in, const matrix_t& gd, matrix_t& co_gd)
        {
                const size_t crows = static_cast<size_t>(co_gd.rows());
                const size_t ccols = static_cast<size_t>(co_gd.cols());

                const size_t orows = static_cast<size_t>(in.rows() - crows);
                const size_t ocols = static_cast<size_t>(in.cols() - ccols);

                for (size_t r = 0; r < crows; r ++)
                {
                        for (size_t c = 0; c < ccols; c ++)
                        {
                                co_gd(r, c) += in.block(r, c, orows, ocols).cwiseProduct(gd).sum();
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        static void backward(const matrix_t& gd, const matrix_t& co, matrix_t& in_gd)
        {
                const size_t crows = static_cast<size_t>(co.rows());
                const size_t ccols = static_cast<size_t>(co.cols());

                const size_t orows = static_cast<size_t>(gd.rows());
                const size_t ocols = static_cast<size_t>(gd.cols());

                for (size_t r = 0; r < orows; r ++)
                {
                        for (size_t c = 0; c < ocols; c ++)
                        {
                                // FIXME: can this be written more efficiently as a block operation?!
                                for (size_t rr = 0; rr < crows; rr ++)
                                {
                                        for (size_t cc = 0; cc < ccols; cc ++)
                                        {
                                                in_gd(r + rr, c + cc) += gd(r, c) * co(rr, cc);
                                        }
                                }
                        }
                }
        }

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
                for (matrix_t& mat : m_data)
                {
                        mat.setZero();
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

        tensor3d_t tensor4d_t::forward(const tensor3d_t& input) const
        {
                assert(n_dim1() == input.n_dim1());
                assert(n_rows()  < input.n_rows());
                assert(n_cols()  < input.n_cols());

                const size_t orows = input.n_rows() - n_rows();
                const size_t ocols = input.n_cols() - n_cols();

                tensor3d_t result(n_dim2(), orows, ocols);

                for (size_t d2 = 0; d2 < n_dim2(); d2 ++)
                {
                        for (size_t d1 = 0; d1 < n_dim1(); d1 ++)
                        {
                                ncv::forward(input.data(d1), data(d1, d2), result.data(d2));
                        }
                }

                return result;
        }

        //-------------------------------------------------------------------------------------------------

        tensor3d_t tensor4d_t::backward(const tensor3d_t& gradient) const
        {
                assert(n_dim2() == gradient.n_dim2());
                assert(0         < gradient.n_rows());
                assert(0         < gradient.n_cols());

                tensor3d_t result(n_dim1(), n_rows(), n_cols());

                for (size_t d1 = 0; d1 < n_dim1(); d1 ++)
                {
                        for (size_t d2 = 0; d2 < n_dim2(); d2 ++)
                        {
                                ncv::backward(gradient.data(d2), data(d1, d2), result.data(d1));
                        }
                }

                return tensor3d_t();
        }

        //-------------------------------------------------------------------------------------------------

        tensor4d_t tensor4d_t::gradient(const tensor3d_t& input, const tensor3d_t& gradient) const
        {
                assert(n_dim1() == input.n_dim1());
                assert(n_dim2() == gradient.n_dim1());
                assert(n_rows()  < input.n_rows());
                assert(n_cols()  < input.n_cols());

                assert(n_rows() + gradient.n_rows() == input.n_rows());
                assert(n_cols() + gradient.n_cols() == input.n_cols());

                tensor4d_t result(n_dim1(), n_dim2(), n_rows(), n_cols());

                for (size_t d1 = 0; d1 < n_dim1(); d1 ++)
                {
                        for (size_t d2 = 0; d2 < n_dim2(); d2 ++)
                        {
                                ncv::gradient(input.data(d1), gradient.data(d2), result.data(d1, d2));
                        }
                }

                return tensor4d_t();
        }

        //-------------------------------------------------------------------------------------------------

        void tensor4d_t::serialize(serializer_t& s) const
        {
                for (size_t d1 = 0; d1 < n_dim1(); d1 ++)
                {
                        s << m_data[d1];
                }
        }

        //-------------------------------------------------------------------------------------------------

        void tensor4d_t::deserialize(deserializer_t& s)
        {
                for (size_t d1 = 0; d1 < n_dim1(); d1 ++)
                {
                        s >> m_data[d1];
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
                                data(d1, d2).noalias() += other.data(d1, d2);
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------
}

