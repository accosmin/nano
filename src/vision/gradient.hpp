#pragma once

#include <cassert>
#include "math/numeric.hpp"

namespace nano
{
        ///
        /// \brief compute the x gradient of the input matrix
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixo
        >
        void gradientx(const tmatrixi& srcplane, tmatrixo&& xplane)
        {
                assert(srcplane.rows() == xplane.rows());
                assert(srcplane.cols() == xplane.cols());

                const int rows = static_cast<int>(srcplane.rows());
                const int cols = static_cast<int>(srcplane.cols());

                for (int r = 0; r < rows; ++ r)
                {
                        for (int c = 0; c < cols; ++ c)
                        {
                                const int cn = nano::clamp(c - 1, 0, cols - 1);
                                const int cp = nano::clamp(c + 1, 0, cols - 1);

                                xplane(r, c) = srcplane(r, cp) - srcplane(r, cn);
                        }
                }
        }

        ///
        /// \brief compute the y gradient of the input matrix
        ///
        template
        <
                typename tmatrixi,
                typename tmatrixo
        >
        void gradienty(const tmatrixi& srcplane, tmatrixo&& yplane)
        {
                assert(srcplane.rows() == yplane.rows());
                assert(srcplane.cols() == yplane.cols());

                const int rows = static_cast<int>(srcplane.rows());
                const int cols = static_cast<int>(srcplane.cols());

                for (int r = 0; r < rows; ++ r)
                {
                        const int rn = nano::clamp(r - 1, 0, rows - 1);
                        const int rp = nano::clamp(r + 1, 0, rows - 1);

                        for (int c = 0; c < cols; ++ c)
                        {
                                yplane(r, c) = srcplane(rp, c) - srcplane(rn, c);
                        }
                }
        }
}

