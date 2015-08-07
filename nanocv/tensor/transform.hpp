#pragma once

#include <algorithm>

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief transform coefficient-wise a tensor: out = op(in)
                ///
                template
                <
                        typename tin_tensor,
                        typename tout_tensor,
                        typename toperator
                >
                void transform(const tin_tensor& in, tout_tensor& out, toperator op)
                {
                        std::transform(in.data(), in.data() + in.size(), out.data(), op);
                }

                ///
                /// \brief transform coefficient-wise a tensor: out = op(in1, in2)
                ///
                template
                <
                        typename tin1_tensor,
                        typename tin2_tensor,
                        typename tout_tensor,
                        typename toperator
                >
                void transform(const tin1_tensor& in1, const tin2_tensor& in2, tout_tensor& out, toperator op)
                {
                        std::transform(in1.data(), in1.data() + in1.size(), in2.data(), out.data(), op);
                }

                ///
                /// \brief transform coefficient-wise a tensor: out = op(in1, in2, in3)
                ///
                template
                <
                        typename tin1_tensor,
                        typename tin2_tensor,
                        typename tin3_tensor,
                        typename tout_tensor,
                        typename toperator
                >
                void transform(const tin1_tensor& in1, const tin2_tensor& in2, const tin3_tensor& in3,
                        tout_tensor& out, toperator op)
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
