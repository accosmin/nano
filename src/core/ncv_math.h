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
        // numerical utility functions.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                // force a value in a given range
                using boost::algorithm::clamp;
                using boost::algorithm::clamp_range;

                // usefull constants
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

                // safely invert a value
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

                // return the sign of a value
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

                // transform coefficient-wise a matrix: out = op(in)
                template
                <
                        typename tmatrix,
                        typename toperator
                >
                void foreach(const tmatrix& src, tmatrix& dst, const toperator& op)
                {
                        typedef typename tmatrix::Scalar tvalue;

                        dst.resize(src.rows(), src.cols());
                        std::transform(src.data(), src.data() + src.size(), dst.data(),
                                       [&op] (tvalue i) { return cast<tvalue>(op(i)); });
                }

                // transform coefficient-wise a matrix: out = op(in, out)
                template
                <
                        typename tmatrix,
                        typename toperator
                >
                void transform(const tmatrix& src, tmatrix& dst, const toperator& op)
                {
                        typedef typename tmatrix::Scalar tvalue;

                        std::transform(src.data(), src.data() + src.size(), dst.data(), dst.data(),
                                       [&op] (tvalue i, tvalue o) { return cast<tvalue>(op(i, o)); });
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

                // scale the input matrix by the given factor (using bilinear interpolation)
                template
                <       typename tmatrix,
                        typename tscalar = double
                >
                void scale(const tmatrix& src, tmatrix& dst, tscalar factor)
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

                // scale the input matrix to the given maximum size
                template
                <
                        typename tmatrix,
                        typename tsize = int
                >
                void scale(const tmatrix& src, tmatrix& dst, tsize max_rows, tsize max_cols)
                {
                        typedef typename tmatrix::Scalar tvalue;

                        const tsize rows = cast<tsize>(src.rows());
                        const tsize cols = cast<tsize>(src.cols());

                        if (    rows < plus_one<tsize>() || max_rows < plus_one<tsize>() ||
                                cols < plus_one<tsize>() || max_cols < plus_one<tsize>())
                        {
                                dst = src;
                        }
                        else
                        {
                                const double srows = cast<double>(max_rows) / cast<double>(rows);
                                const double scols = cast<double>(max_cols) / cast<double>(cols);
                                scale(src, dst, std::min(srows, scols));
                        }
                }
        }
}

// serialize matrices and vectors
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

