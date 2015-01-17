#pragma once

#include "random.hpp"
#include "bilinear.hpp"

namespace ncv
{
        ///
        /// \brief in-place random translate [-range, +range]
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
                const double scalex = math::cast<double>(cols + math::abs(dx) + 1) / math::cast<double>(cols);
                const double scaley = math::cast<double>(rows + math::abs(dy) + 1) / math::cast<double>(rows);

                tmatrix scaled_src;
                bilinear(src, scaled_src, std::max(scalex, scaley), mixer);


//                // construct Gaussian kernel
//                const std::vector<tscalar> kernel = make_gaussian(sigma);

//                const int ksize = static_cast<int>(kernel.size());
//                const int krad = ksize / 2;

//                if (ksize != (2 * krad + 1))
//                {
//                        return false;
//                }

//                std::vector<tscalar> buff(std::max(rows, cols));

//                // horizontal filter
//                for (int r = 0; r < rows; r ++)
//                {
//                        for (int c = 0; c < cols; c ++)
//                        {
//                                buff[c] = math::cast<tscalar>(getter(src(r, c)));
//                        }

//                        for (int c = 0; c < cols; c ++)
//                        {
//                                tscalar v = 0;
//                                for (int k = -krad; k <= krad; k ++)
//                                {
//                                        const int cc = math::clamp(k + c, 0, cols - 1);
//                                        v += kernel[k + krad] * buff[cc];
//                                }

//                                src(r, c) = setter(src(r, c), math::cast<tvalue>(math::clamp(v, minv, maxv)));
//                        }
//                }

//                // vertical filter
//                for (int c = 0; c < cols; c ++)
//                {
//                        for (int r = 0; r < rows; r ++)
//                        {
//                                buff[r] = math::cast<tscalar>(getter(src(r, c)));
//                        }

//                        for (int r = 0; r < rows; r ++)
//                        {
//                                tscalar v = 0;
//                                for (int k = -krad; k <= krad; k ++)
//                                {
//                                        const int rr = math::clamp(k + r, 0, rows - 1);
//                                        v += kernel[k + krad] * buff[rr];
//                                }

//                                src(r, c) = setter(src(r, c), math::cast<tvalue>(math::clamp(v, minv, maxv)));
//                        }
//                }

                return true;
        }
}

