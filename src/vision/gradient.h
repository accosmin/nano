#pragma once

#include "tensor.h"
#include "math/numeric.h"

namespace nano
{
        ///
        /// \brief compute the y gradient of the input matrix.
        ///
        template <typename tmatrixi>
        matrix_t gradienty(const tmatrixi& src)
        {
                const auto rows = src.rows();
                const auto cols = src.cols();

                matrix_t grad(rows, cols);
                for (tensor_size_t r = 0; r < rows; ++ r)
                {
                        const auto rn = nano::clamp(r - 1, 0, rows - 1);
                        const auto rp = nano::clamp(r + 1, 0, rows - 1);

                        grad.row(r) = src.row(rp) - src.row(rn);
                }

                return grad;
        }
        ///
        /// \brief compute the x gradient of the input matrix.
        ///
        template <typename tmatrixi>
        matrix_t gradientx(const tmatrixi& src)
        {
                matrix_t tsrc = src.transpose();
                matrix_t grad = gradienty(tsrc);
                grad.transposeInPlace();
                return grad;
        }
}
