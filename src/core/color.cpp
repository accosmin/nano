#include "color.h"
#include "math/math.hpp"

namespace ncv
{
        namespace impl
        {
                // note: the RGB - XYZ - CIELab color transformations are taken from:
                //      --- http://www.easyrgb.com/ ---                

                /////////////////////////////////////////////////////////////////////////////////////////

                scalar_t fn_pow_12_5(scalar_t x)
                {
                        return std::pow(x, 2.4);
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                scalar_t fn_pow_5_12(scalar_t x)
                {
                        // 5/12 = 1/3 + 1/2 * 1/2 * 1/3;
                        const scalar_t cbx = std::cbrt(x);
                        return cbx * std::sqrt(std::sqrt(cbx));
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                scalar_t fn_xyz2lab(scalar_t t)
                {
                        if (t > 0.008856)
                        {
                                return std::cbrt(t);
                        }
                        else
                        {
                                return 7.787 * t + 16.0 / 116.0;
                        }
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                scalar_t fn_lab2xyz(scalar_t t)
                {
                        static const scalar_t thres = std::cbrt(0.008856);

                        if (t > thres)
                        {
                                return t * t * t;
                        }
                        else
                        {
                                return (t - 16.0 / 116.0) / 7.787;
                        }
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                scalar_t fn_rgb2xyz(scalar_t t)
                {
                        if (t > 0.04045)
                        {
                                return fn_pow_12_5((t + 0.055 ) / 1.055);
                        }
                        else
                        {
                                return t / 12.92;
                        }
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                scalar_t fn_xyz2rgb(scalar_t t)
                {
                        if (t > 0.0031308)
                        {
                                return 1.055 * fn_pow_5_12(t) - 0.055;
                        }
                        else
                        {
                                return 12.92 * t;
                        }
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                class rgb2xyz_map
                {
                public:

                        // constructor
                        rgb2xyz_map()
                        {
                                for (rgba_t rgb = 0; rgb < 256; rgb ++)
                                {
                                        const scalar_t var = fn_rgb2xyz(rgb / 255.0) * 100.0;

                                        m_r2xs[rgb] = var * 0.4124;
                                        m_g2xs[rgb] = var * 0.3576;
                                        m_b2xs[rgb] = var * 0.1805;

                                        m_r2ys[rgb] = var * 0.2126;
                                        m_g2ys[rgb] = var * 0.7152;
                                        m_b2ys[rgb] = var * 0.0722;

                                        m_r2zs[rgb] = var * 0.0193;
                                        m_g2zs[rgb] = var * 0.1192;
                                        m_b2zs[rgb] = var * 0.9505;

                                }
                        }

                        //
                        void operator()(rgba_t rgb_r, rgba_t rgb_g, rgba_t rgb_b,
                                        scalar_t& xyz_x, scalar_t& xyz_y, scalar_t& xyz_z) const
                        {
                                xyz_x = m_r2xs[rgb_r] + m_g2xs[rgb_g] + m_b2xs[rgb_b];
                                xyz_y = m_r2ys[rgb_r] + m_g2ys[rgb_g] + m_b2ys[rgb_b];
                                xyz_z = m_r2zs[rgb_r] + m_g2zs[rgb_g] + m_b2zs[rgb_b];
                        }

                private:

                        // attributes
                        scalar_t        m_r2xs[256];
                        scalar_t        m_r2ys[256];
                        scalar_t        m_r2zs[256];

                        scalar_t        m_g2xs[256];
                        scalar_t        m_g2ys[256];
                        scalar_t        m_g2zs[256];

                        scalar_t        m_b2xs[256];
                        scalar_t        m_b2ys[256];
                        scalar_t        m_b2zs[256];
                };

                static const rgb2xyz_map the_rgb2xyz_map;

                /////////////////////////////////////////////////////////////////////////////////////////

                void rgb2xyz(rgba_t rgb_r, rgba_t rgb_g, rgba_t rgb_b,
                             scalar_t& xyz_x, scalar_t& xyz_y, scalar_t& xyz_z)
                {
                        the_rgb2xyz_map(rgb_r, rgb_g, rgb_b, xyz_x, xyz_y, xyz_z);
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                void xyz2rgb(scalar_t xyz_x, scalar_t xyz_y, scalar_t xyz_z,
                             rgba_t& rgb_r, rgba_t& rgb_g, rgba_t& rgb_b)
                {
                        const scalar_t var_r = (xyz_x *  3.2406 + xyz_y * -1.5372 + xyz_z * -0.4986) / 100.0;
                        const scalar_t var_g = (xyz_x * -0.9689 + xyz_y *  1.8758 + xyz_z *  0.0415) / 100.0;
                        const scalar_t var_b = (xyz_x *  0.0557 + xyz_y * -0.2040 + xyz_z *  1.0570) / 100.0;

                        rgb_r = math::clamp(math::cast<rgba_t>(255.0 * fn_xyz2rgb(var_r)), 0, 255);
                        rgb_g = math::clamp(math::cast<rgba_t>(255.0 * fn_xyz2rgb(var_g)), 0, 255);
                        rgb_b = math::clamp(math::cast<rgba_t>(255.0 * fn_xyz2rgb(var_b)), 0, 255);
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                void xyz2lab(scalar_t xyz_x, scalar_t xyz_y, scalar_t xyz_z,
                             scalar_t& cie_l, scalar_t& cie_a, scalar_t& cie_b)
                {
                        // Observer. = 2°, Illuminant = D65
                        static const scalar_t x_n = 95.047;
                        static const scalar_t y_n = 100.000;
                        static const scalar_t z_n = 108.883;

                        const scalar_t var_x = fn_xyz2lab(xyz_x / x_n);
                        const scalar_t var_y = fn_xyz2lab(xyz_y / y_n);
                        const scalar_t var_z = fn_xyz2lab(xyz_z / z_n);

                        cie_l = 116.0 * var_y - 16.0;
                        cie_a = 500.0 * (var_x - var_y);
                        cie_b = 200.0 * (var_y - var_z);
                }

                /////////////////////////////////////////////////////////////////////////////////////////

                void lab2xyz(scalar_t cie_l, scalar_t cie_a, scalar_t cie_b,
                             scalar_t& xyz_x, scalar_t& xyz_y, scalar_t& xyz_z)
                {
                        // Observer. = 2°, Illuminant = D65
                        static const scalar_t x_n = 95.047;
                        static const scalar_t y_n = 100.000;
                        static const scalar_t z_n = 108.883;

                        // CIELab to XYZ
                        const scalar_t var_y = (cie_l + 16.0) / 116.0;
                        const scalar_t var_x = cie_a / 500.0 + var_y;
                        const scalar_t var_z = var_y - cie_b / 200.0;

                        xyz_x = x_n * fn_lab2xyz(var_x);
                        xyz_y = y_n * fn_lab2xyz(var_y);
                        xyz_z = z_n * fn_lab2xyz(var_z);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        cielab_t color::make_cielab(rgba_t rgba)
        {
                cielab_t cielab;
                scalar_t x, y, z;

                impl::rgb2xyz(make_red(rgba), make_green(rgba), make_blue(rgba), x, y, z);
                impl::xyz2lab(x, y, z, cielab(0), cielab(1), cielab(2));

                return cielab;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        rgba_t color::make_rgba(const cielab_t& cielab)
        {
                rgba_t r, g, b;
                scalar_t x, y, z;

                impl::lab2xyz(cielab(0), cielab(1), cielab(2), x, y, z);
                impl::xyz2rgb(x, y, z, r, g, b);

                return make_rgba(r, g, b);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        rgba_matrix_t color::make_rgba(const matrix_t& data)
        {
                return make_rgba(data, data.minCoeff(), data.maxCoeff());
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        rgba_matrix_t color::make_rgba(const matrix_t& data, scalar_t min_, scalar_t max_)
        {
                const scalar_t min = std::min(min_, max_) - std::numeric_limits<scalar_t>::epsilon();
                const scalar_t max = std::max(min_, max_) + std::numeric_limits<scalar_t>::epsilon();
                const scalar_t delta = std::fabs(max - min);
                const scalar_t scale = 255.0 / delta;

                const int rows = static_cast<int>(data.rows());
                const int cols = static_cast<int>(data.cols());
                const int size = rows * cols;

                rgba_matrix_t rgba(rows, cols);
                for (int i = 0; i < size; i ++)
                {
                        const scalar_t nval = scale * (math::clamp(data(i), min, max) - min);
                        const rgba_t gray = math::cast<rgba_t>(nval);
                        rgba(i) = color::make_rgba(gray, gray, gray, 255);
                }

                return rgba;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
