#ifndef NANOCV_MATH_H
#define NANOCV_MATH_H

#include <algorithm>
#include <type_traits>
#include <boost/algorithm/clamp.hpp>
#include <boost/math/constants/constants.hpp>
#include "ncv_types.h"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Numerical utility functions.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace math
{
        // Force a value in a given range
        using boost::algorithm::clamp;
        using boost::algorithm::clamp_range;

        // Usefull constants
        using boost::math::constants::pi;

        template
        <
                typename tvalue
        >
        tvalue zero()
        {
                return static_cast<tvalue>(0);
        }

        template
        <
                typename tvalue
        >
        tvalue half()
        {
                return static_cast<tvalue>(0.5);
        }

        template
        <
                typename tvalue
        >
        tvalue plus_one()
        {
                return static_cast<tvalue>(+1);
        }

        template
        <
                typename tvalue
        >
        tvalue minus_one()
        {
                return static_cast<tvalue>(-1);
        }

        // Safely invert a value
        template
        <
                typename tvalue,
                typename tscalar = double
        >
        tscalar inverse(tvalue value)
        {
                return  std::abs(value) > std::numeric_limits<tvalue>::epsilon() ?
                        plus_one<tscalar>() / value : plus_one<tscalar>();
        }

        // Return the sign of a value
        template
        <
                typename tscalar = double
        >
        inline tscalar sign(tscalar value)
        {
                return  value > zero<tscalar>() ? plus_one<tscalar>() :
                        (value < zero<tscalar>() ? minus_one<tscalar>() : zero<tscalar>());
        }

        namespace impl
        {
                template
                <
                        typename tround,
                        bool tround_integral,
                        typename tvalue,
                        bool tvalue_integral
                >
                struct cast
                {
                        static tround dispatch(tvalue value)
                        {
                                return static_cast<tround>(value);
                        }
                };

                template
                <
                        typename tround,
                        typename tvalue
                >
                struct cast<tround, true, tvalue, false>
                {
                        static tround dispatch(tvalue value)
                        {
                                return  static_cast<tround>(
                                        value < zero<tvalue>() ? value - half<tvalue>() :
                                        (value > zero<tvalue>() ? value + half<tvalue>() : value));
                        }
                };
        }

        // Cast a value to another with rounding to the cloasest if necessary
        template
        <
                typename tround,
                typename tvalue
        >
        tround cast(tvalue value)
        {
                return  impl::cast<
                        tround, std::is_integral<tround>::value,
                        tvalue, std::is_integral<tvalue>::value>::dispatch(value);
        }

        // Cast (with rounding if necessary) a matrix type to another one
        template
        <
                typename tsrc_value,
                typename tdst_value
        >
        void cast(
                const typename matrix<tsrc_value>::matrix_t& in,
                typename matrix<tdst_value>::matrix_t& out)
        {
                out.resize(in.rows(), in.cols());
                std::transform(in.data(), in.data() + in.size(), out.data(), [] (tsrc_value i)
                {
                        return cast<tdst_value>(i);
                });
        }

        // Transform coefficient-wise a matrix: out = op(in)
        template
        <
                typename tvalue,
                typename toperator
        >
        void foreach(
                const typename matrix<tvalue>::matrix_t& in,
                const toperator& op,
                typename matrix<tvalue>::matrix_t& out)
        {
                out.resize(in.rows(), in.cols());
                std::transform(in.data(), in.data() + in.size(), out.data(), [&op] (tvalue i)
                {
                        return cast<tvalue>(op(i));
                });
        }

        // Transform coefficient-wise a matrix: out = op(in, out)
        template
        <
                typename tvalue,
                typename toperator
        >
        void transform(
                const typename matrix<tvalue>::matrix_t& in,
                const toperator& op,
                typename matrix<tvalue>::matrix_t& out)
        {
                std::transform(in.data(), in.data() + in.size(), out.data(), out.data(), [&op] (tvalue i, tvalue o)
                {
                        return cast<tvalue>(op(i, o));
                });
        }

        // Normalize the input matrix from [imin, imax] to [omin, omax] output matrix
        template
        <
                typename tsrc_value,
                typename tdst_value,
                typename tscalar = double
        >
        void norm(
                const typename matrix<tsrc_value>::matrix_t& in,
                tscalar imin, tscalar imax, tscalar omin, tscalar omax,
                typename matrix<tdst_value>::matrix_t& out)
        {
                static const tscalar eps = cast<tscalar>(1e-6);

                if (imin + eps < imax)
                {
                        const tscalar s = (omax - omin) / (imax - imin);

                        out.resize(in.rows(), in.cols());
                        std::transform(in.data(), in.data() + in.size(), out.data(), [=] (tsrc_value v)
                        {
                                return cast<tdst_value>(omin + s * (cast<tscalar>(v) - imin));
                        });
                }
                else
                {
                        cast(in, out);
                }
        }

        // Normalize the input matrix to [omin, omax] output matrix
        template
        <
                typename tsrc_value,
                typename tdst_value,
                typename tscalar = double
        >
        void norm(
                const typename matrix<tsrc_value>::matrix_t& in,
                tscalar omin, tscalar omax,
                typename matrix<tdst_value>::matrix_t& out)
        {
                norm<tsrc_value, tdst_value, tscalar>(
                        in, cast<tscalar>(in.minCoeff()), cast<tscalar>(in.maxCoeff()), omin, omax, out);
        }

        // Scale the input matrix by the given factor
        // (using bilinear interpolation)
        template
        <       typename tvalue,
                typename tscalar = double
        >
        void scale(
                const typename matrix<tvalue>::matrix_t& in,
                tscalar factor,
                typename matrix<tvalue>::matrix_t& out)
        {
                static const tscalar eps = cast<tscalar>(1e-2);
                static const tscalar one = cast<tscalar>(1.0);

                if (factor < eps || std::abs(factor - one) < eps)
                {
                        out = in;
                }
                else
                {
                        const int irows = cast<int>(in.rows());
                        const int icols = cast<int>(in.cols());
                        const int orows = cast<int>(factor * irows);
                        const int ocols = cast<int>(factor * icols);
                        const tscalar is = one / factor;

                        // Bilinear interpolation
                        out.resize(orows, ocols);
                        for (int _or = 0; _or < orows; _or ++)
                        {
                                const tscalar isr = is * _or;
                                const int ir0 = cast<int>(isr), ir1 = std::min(ir0 + 1, irows - 1);
                                const tscalar wr1 = isr - ir0, wr0 = one - wr1;

                                for (int _oc = 0; _oc < ocols; _oc ++)
                                {
                                        const tscalar isc = is * _oc;
                                        const int ic0 = cast<int>(isc), ic1 = std::min(ic0 + 1, icols - 1);
                                        const tscalar wc1 = isc - ic0, wc0 = one - wc1;

                                        out(_or, _oc) = cast<tvalue>(
                                                wr0 * wc0 * in(ir0, ic0) +
                                                wr0 * wc1 * in(ir0, ic1) +
                                                wr1 * wc1 * in(ir1, ic1) +
                                                wr1 * wc0 * in(ir1, ic0));
                                }
                        }
                }
        }

        // Scale the input matrix to the given maximum size
        template
        <
                typename tvalue,
                typename tsize = int
        >
        void scale(
                const typename matrix<tvalue>::matrix_t& in,
                tsize max_rows, tsize max_cols,
                typename matrix<tvalue>::matrix_t& out)
        {
                const tsize rows = cast<tsize>(in.rows());
                const tsize cols = cast<tsize>(in.cols());

                if (    rows < plus_one<tsize>() || max_rows < plus_one<tsize>() ||
                        cols < plus_one<tsize>() || max_cols < plus_one<tsize>())
                {
                        out = in;
                }
                else
                {
                        const double srows = cast<double>(max_rows) / cast<double>(rows);
                        const double scols = cast<double>(max_cols) / cast<double>(cols);
                        scale<tvalue>(in, std::min(srows, scols), out);
                }
        }

        // 1D element-wise (dot) product
        template
        <
                typename tvalue
        >
        tvalue dot(
                const typename vector<tvalue>::vector_t& k1,
                const typename vector<tvalue>::vector_t& k2)
        {
                return k1.dot(k2);
        }

        // 2D element-wise (Hadamard) product
        template
        <
                typename tvalue
        >
        tvalue dot(
                const typename matrix<tvalue>::matrix_t& k1,
                const typename matrix<tvalue>::matrix_t& k2)
        {
                return k1.cwiseProduct(k2).sum();
        }

        // 1D horizontal convolution of the input matrix to output matrix: out = op(in)
        template
        <
                typename tvalue,
                typename toperator
        >
        void conv(
                const typename matrix<tvalue>::matrix_t& in,
                int ksize, toperator op,
                typename matrix<tvalue>::matrix_t& out)
        {
                const int rows = cast<int>(in.rows());
                const int cols = cast<int>(in.cols());
                const int cmin = 0, cmax = cols - 1;

                typename vector<tvalue>::vector_t data(ksize);
                const int kmin = - ksize / 2, kmax = kmin + ksize;

                out.resize(rows, cols);
                for (int r = 0; r < rows; r ++)
                {
                        for (int c = 0; c < cols; c ++)
                        {
                                for (int k = kmin, kk = 0; k < kmax; k ++, kk ++)
                                {
                                        const int _kr = r, _kc = clamp(c + k, cmin, cmax);
                                        data(kk) = in(_kr, _kc);
                                }

                                out(r, c) = op(data);
                        }
                }
        }

        // 1D horizontal convolution of the input matrix to output matrix: out = in * kernel
        template
        <
                typename tvalue
        >
        void conv(
                const typename matrix<tvalue>::matrix_t& in,
                const typename vector<tvalue>::vector_t& kernel,
                typename matrix<tvalue>::matrix_t& out)
        {
                const int ksize = cast<int>(kernel.rows());

                conv<tvalue>(
                        in, ksize,
                        [&](const typename vector<tvalue>::vector_t& data) { return dot<tvalue>(data, kernel); },
                        out);
        }

        // 2D convolution of the input matrix to output matrix: out = op(in)
        template
        <
                typename tvalue,
                typename toperator
        >
        void conv(
                const typename matrix<tvalue>::matrix_t& in,
                int krows, int kcols, const toperator& op,
                typename matrix<tvalue>::matrix_t& out)
        {
                const int rows = cast<int>(in.rows());
                const int cols = cast<int>(in.cols());
                const int cmin = 0, cmax = cols - 1;
                const int rmin = 0, rmax = rows - 1;

                typename matrix<tvalue>::matrix_t data(krows, kcols);
                const int krmin = - krows / 2, krmax = krmin + krows;
                const int kcmin = - kcols / 2, kcmax = kcmin + kcols;

                out.resize(rows, cols);
                for (int r = 0; r < rows; r ++)
                {
                        for (int c = 0; c < cols; c ++)
                        {
                                for (int kr = krmin, kkr = 0; kr < krmax; kr ++, kkr ++)
                                {
                                        const int _kr = clamp(r + kr, rmin, rmax);
                                        for (int kc = kcmin, kkc = 0; kc < kcmax; kc ++, kkc ++)
                                        {
                                                const int _kc = clamp(c + kc, cmin, cmax);
                                                data(kkr, kkc) = in(_kr, _kc);
                                        }
                                }

                                out(r, c) = op(data);
                        }
                }
        }

        // 2D convolution of the input matrix to output matrix: out = in * kernel
        template
        <
                typename tvalue
        >
        void conv(
                const typename matrix<tvalue>::matrix_t& in,
                const typename matrix<tvalue>::matrix_t& kernel,
                typename matrix<tvalue>::matrix_t& out)
        {
                const int krows = cast<int>(kernel.rows());
                const int kcols = cast<int>(kernel.cols());

                conv<tvalue>(
                        in, krows, kcols,
                        [&](const typename matrix<tvalue>::matrix_t& data) { return dot<tvalue>(data, kernel); },
                        out);
        }

        // Save/load vectors/matrices to/from arrays
        template
        <
                typename tvalue
        >
        inline tvalue* vec2array(const typename vector<tvalue>::vector_t& v, tvalue* x)
        {
                std::copy(v.data(), v.data() + v.size(), x);
                return (x += v.size());
        }
        template
        <
                typename tvalue
        >
        inline tvalue* mat2array(const typename matrix<tvalue>::matrix_t& m, tvalue* x)
        {
                std::copy(m.data(), m.data() + m.size(), x);
                return (x += m.size());
        }
        template
        <
                typename tvalue
        >
        inline const tvalue* array2vec(const tvalue* x, typename vector<tvalue>::vector_t& v)
        {
                std::copy(x, x + v.size(), v.data());
                return (x += v.size());
        }
        template
        <
                typename tvalue
        >
        inline const tvalue* array2mat(const tvalue* x, typename matrix<tvalue>::matrix_t& m)
        {
                std::copy(x, x + m.size(), m.data());
                return (x += m.size());
        }
}
}

// Serialize matrices and vectors
namespace boost
{
namespace serialization
{
        template
        <
                class tarchive,
                class tvalue,
                int Rows,
                int Cols,
                int Options
        >
        void serialize(tarchive& ar, Eigen::Matrix<tvalue, Rows, Cols, Options>& mat, const unsigned int)
        {
                if (tarchive::is_saving::value)
                {
                        int rows = mat.rows(), cols = mat.cols();
                        ar & rows; ar & cols;

                        for (int i = 0; i < mat.size(); i ++)
                        {
                                ar & mat(i);
                        }
                }

                else
                {
                        int rows, cols;
                        ar & rows; ar & cols;

                        mat.resize(rows, cols);
                        for (int i = 0; i < mat.size(); i ++)
                        {
                                ar & mat(i);
                        }
                }
        }
}
}

#endif // NANOCV_MATH_H

