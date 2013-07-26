#ifndef NANOCV_MATRIX_ALGO_H
#define NANOCV_MATRIX_ALGO_H

#include <algorithm>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // numerical utility functions for matrices.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
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
        }
}

#endif // NANOCV_MATRIX_ALGO_H

