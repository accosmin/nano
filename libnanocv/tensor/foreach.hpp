#pragma once

#include <algorithm>

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief in-place transform coefficient-wise a matrix: op(in)
                ///
                template
                <
                        typename tmatrix,
                        typename toperator
                >
                void for_each(tmatrix& src, toperator op)
                {
                        std::for_each(src.data(), src.data() + src.size(), op);
                }
        }
}
