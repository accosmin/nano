#ifndef NANOCV_TRANSFORM_H
#define NANOCV_TRANSFORM_H

#include <algorithm>

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // numerical utility functions for matrices.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                // transform coefficient-wise a matrix: out = op(in)
                template
                <
                        typename tin_matrix,
                        typename tout_matrix,
                        typename toperator
                >
                void transform(const tin_matrix& in, tout_matrix& out, toperator op)
                {
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
                        auto in1_it = in1.data(), in1_end = in1.data() + in1.size();
                        auto in2_it = in2.data();
                        auto in3_it = in3.data();
                        auto out_it = out.data();

                        for ( ; in1_it != in1_end; ++ in1_it, ++ in2_it, ++ in3_it, ++ out_it)
                        {
                                *out_it = op(*in1_it, *in2_it, *in3_it);
                        }
                }
        }
}

#endif // NANOCV_TRANSFORM_H

