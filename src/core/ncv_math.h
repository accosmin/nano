#ifndef NANOCV_MATH_H
#define NANOCV_MATH_H

#include <algorithm>
#include <type_traits>
#include <boost/algorithm/clamp.hpp>
#include <boost/math/constants/constants.hpp>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // numerical utility functions.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                // force a value in a given range
                using boost::algorithm::clamp;
                using boost::algorithm::clamp_range;

                // usefull constants
                using boost::math::constants::pi;
                using namespace boost::math::constants;

                // square a value
                template
                <
                        typename tvalue
                >
                tvalue square(tvalue value)
                {
                        return value * value;
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
                                        return static_cast<tround>(std::nearbyint(value));
                                }
                        };
                }

                // cast a value to another type (with rounding to the closest if necessary)
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

                // transform coefficient-wise a matrix: op(&in)
                template
                <
                        typename tmatrix,
                        typename toperator
                >
                void for_each(tmatrix& in, toperator op)
                {
                        std::for_each(in.data(), in.data() + in.size(), op);
                }

                // transform coefficient-wise a matrix: out = op(in)
                template
                <
                        typename tin_matrix,
                        typename tout_matrix,
                        typename toperator
                >
                void transform(const tin_matrix& in, tout_matrix& out, toperator op)
                {
                        out.resize(in.rows(), in.cols());
                        std::transform(in.data(), in.data() + in.size(), out.data(), op);
                }

                // transform coefficient-wise a matrix: out = op(in1, in2)
                template
                <
                        typename tin1_matrix,
                        typename tin2_matrix,
                        typename tout_matrix,
                        typename toperator
                >
                void transform(const tin1_matrix& in1, const tin2_matrix& in2, tout_matrix& out, toperator op)
                {
                        out.resize(in1.rows(), in1.cols());
                        std::transform(in1.data(), in1.data() + in1.size(), in2.data(), out.data(), op);
                }

                // transform coefficient-wise a matrix: out = op(in1, in2, in3)
                template
                <
                        typename tin1_matrix,
                        typename tin2_matrix,
                        typename tin3_matrix,
                        typename tout_matrix,
                        typename toperator
                >
                void transform(const tin1_matrix& in1, const tin2_matrix& in2, const tin3_matrix& in3,
                               tout_matrix& out, toperator op)
                {
                        out.resize(in1.rows(), in1.cols());
                        std::transform(in1.data(), in1.data() + in1.size(), in2.data(), in3.data(), out.data(), op);
                }

                // cast a matrix type to another one
                template
                <
                        typename tsrc_matrix,
                        typename tdst_matrix
                >
                void cast(const tsrc_matrix& src, tdst_matrix& dst)
                {
                        typedef typename tsrc_matrix::Scalar tsrc_value;
                        typedef typename tdst_matrix::Scalar tdst_value;

                        transform(src, dst, [] (tsrc_value i) { return cast<tdst_value>(i); });
                }

                // normalize the input matrix from [imin, imax] to [omin, omax] output matrix
                template
                <
                        typename tsrc_matrix,
                        typename tdst_matrix,
                        typename tscalar = double
                >
                void norm(const tsrc_matrix& src, tdst_matrix& dst,
                          tscalar imin, tscalar imax, tscalar omin, tscalar omax)
                {
                        typedef typename tsrc_matrix::Scalar tsrc_value;
                        typedef typename tdst_matrix::Scalar tdst_value;

                        static const tscalar eps = cast<tscalar>(1e-6);

                        if (imin + eps < imax)
                        {
                                const tscalar s = (omax - omin) / (imax - imin);

                                transform(src, dst, [=] (tsrc_value v)
                                {
                                        return cast<tdst_value>(omin + s * (cast<tscalar>(v) - imin));
                                });
                        }
                        else
                        {
                                cast(src, dst);
                        }
                }

                // normalize the input matrix to [omin, omax] output matrix
                template
                <
                        typename tsrc_matrix,
                        typename tdst_matrix,
                        typename tscalar = double
                >
                void norm(const tsrc_matrix& src, tdst_matrix& dst,
                          tscalar omin, tscalar omax)
                {
                        norm(src, dst, cast<tscalar>(src.minCoeff()), cast<tscalar>(src.maxCoeff()), omin, omax);
                }

                // resize the input matrix by the given factor (using bilinear interpolation)
                template
                <       typename tmatrix,
                        typename tscalar = double
                >
                void bilinear(const tmatrix& src, tmatrix& dst, tscalar factor)
                {
                        typedef typename tmatrix::Scalar tvalue;

                        static const tscalar eps = cast<tscalar>(1e-2);
                        static const tscalar one = cast<tscalar>(1.0);

                        if (factor < eps || std::abs(factor - one) < eps)
                        {
                                dst = src;
                        }
                        else
                        {
                                const int irows = cast<int>(src.rows());
                                const int icols = cast<int>(src.cols());
                                const int orows = cast<int>(factor * irows);
                                const int ocols = cast<int>(factor * icols);
                                const tscalar is = one / factor;

                                // bilinear interpolation
                                dst.resize(orows, ocols);
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

                                                dst(_or, _oc) = cast<tvalue>(
                                                        wr0 * wc0 * src(ir0, ic0) +
                                                        wr0 * wc1 * src(ir0, ic1) +
                                                        wr1 * wc1 * src(ir1, ic1) +
                                                        wr1 * wc0 * src(ir1, ic0));
                                        }
                                }
                        }
                }

                // resize the input matrix to the given maximum matrix size
                template
                <
                        typename tmatrix,
                        typename tsize = int
                >
                void bilinear(const tmatrix& src, tmatrix& dst, tsize max_rows, tsize max_cols)
                {
                        const tsize rows = cast<tsize>(src.rows());
                        const tsize cols = cast<tsize>(src.cols());
                        const tsize one = static_cast<tsize>(1);

                        if (    rows < one || max_rows < one ||
                                cols < one || max_cols < one)
                        {
                                dst = src;
                        }
                        else
                        {
                                const double srows = cast<double>(max_rows) / cast<double>(rows);
                                const double scols = cast<double>(max_cols) / cast<double>(cols);
                                bilinear(src, dst, std::min(srows, scols));
                        }
                }
        }
}

#endif // NANOCV_MATH_H

