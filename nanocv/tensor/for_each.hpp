#pragma once

#include <algorithm>

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief in-place transform coefficient-wise a tensor: op(in)
                ///
                template
                <
                        typename tinout_tensor,
                        typename toperator
                >
                void for_each(tinout_tensor&& inout, toperator op)
                {
                        std::for_each(inout.data(), inout.data() + inout.size(), op);
                }
        }
}
