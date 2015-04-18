#pragma once

#include "bilinear.hpp"
#include "nanocv/math/random.hpp"

namespace ncv
{
        ///
        /// \brief in-place random translate [-range, +range]
        ///
        /// \note keeps the same size, but upscales the input matrix to fit the translation
        ///
        template
        <
                typename tscalar = double,
                typename tmatrix,
                typename tmixer,                        ///< interpolate (pixel) values given weights

                typename tvalue = typename tmatrix::Scalar
        >
        bool random_translate(tmatrix& src, int range, tmixer mixer)
        {
                const int rows = static_cast<int>(src.rows());
                const int cols = static_cast<int>(src.cols());

                // random translations
                random_t<int> rgen(- math::abs(range) - 1, + math::abs(range) + 1);
                const int dx = rgen();
                const int dy = rgen();

                // scale the image to fit the translation
                const double scalex = math::cast<double>(cols + 2 * math::abs(dx) + 1) / math::cast<double>(cols);
                const double scaley = math::cast<double>(rows + 2 * math::abs(dy) + 1) / math::cast<double>(rows);

                tmatrix ssrc;
                bilinear(src, ssrc, std::max(scalex, scaley), mixer);

                // cut the translated area
                const int srows = static_cast<int>(ssrc.rows());
                const int scols = static_cast<int>(ssrc.cols());

                const int x = dx + (scols - cols) / 2;
                const int y = dy + (srows - rows) / 2;
                src = ssrc.block(y, x, rows, cols);

                // OK
                return true;
        }
}

