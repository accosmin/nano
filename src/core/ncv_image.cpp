#include "ncv_image.h"
#include <QImage>

namespace ncv
{
        namespace impl
        {
                // note: the RGB - XYZ - CIELab color transformations are taken from:
                //      --- http://www.easyrgb.com/ ---                

                //-------------------------------------------------------------------------------------------------

//                scalar_t fn_pow_4(scalar_t x)
//                {
//                        const scalar_t x2 = x * x;
//                        return x2 * x2;
//                }

//                scalar_t fn_pow_5(scalar_t x)
//                {
//                        return x * fn_pow_4(x);
//                }

//                scalar_t fn_pow_1_5(scalar_t a)
//                {
//                        scalar_t x = a, last_x = x;
//                        do
//                        {
//                                last_x = x;
//                                x = 0.8 * last_x + 0.2 * a / fn_pow_4(last_x);
//                        }
//                        while (std::fabs(x - last_x) > 1e-8);

//                        return x;
//                }

                scalar_t fn_pow_12_5(scalar_t x)
                {
//                        const scalar_t r = x * fn_pow_1_5(x);
//                        return r * r;
                        return std::pow(x, 2.4);
                }

                //-------------------------------------------------------------------------------------------------

                scalar_t fn_pow_5_12(scalar_t x)
                {
                        // 5/12 = 1/3 + 1/2 * 1/2 * 1/3;
                        const scalar_t cbx = std::cbrt(x);
                        return cbx * std::sqrt(std::sqrt(cbx));
                }

                //-------------------------------------------------------------------------------------------------

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

                //-------------------------------------------------------------------------------------------------

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

                //-------------------------------------------------------------------------------------------------

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

                //-------------------------------------------------------------------------------------------------

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

                //-------------------------------------------------------------------------------------------------

                class rgb2xyz_map
                {
                public:

                        // constructor
                        rgb2xyz_map()
                        {
                                for (rgb_t rgb = 0; rgb < 256; rgb ++)
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
                        void operator()(rgb_t rgb_r, rgb_t rgb_g, rgb_t rgb_b,
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

                //-------------------------------------------------------------------------------------------------

                void rgb2xyz(rgb_t rgb_r, rgb_t rgb_g, rgb_t rgb_b,
                             scalar_t& xyz_x, scalar_t& xyz_y, scalar_t& xyz_z)
                {
                        the_rgb2xyz_map(rgb_r, rgb_g, rgb_b, xyz_x, xyz_y, xyz_z);
                }

                //-------------------------------------------------------------------------------------------------

                void xyz2rgb(scalar_t xyz_x, scalar_t xyz_y, scalar_t xyz_z,
                             rgb_t& rgb_r, rgb_t& rgb_g, rgb_t& rgb_b)
                {
                        const scalar_t var_r = (xyz_x *  3.2406 + xyz_y * -1.5372 + xyz_z * -0.4986) / 100.0;
                        const scalar_t var_g = (xyz_x * -0.9689 + xyz_y *  1.8758 + xyz_z *  0.0415) / 100.0;
                        const scalar_t var_b = (xyz_x *  0.0557 + xyz_y * -0.2040 + xyz_z *  1.0570) / 100.0;

                        rgb_r = math::clamp(math::cast<rgb_t>(255.0 * fn_xyz2rgb(var_r)), 0, 255);
                        rgb_g = math::clamp(math::cast<rgb_t>(255.0 * fn_xyz2rgb(var_g)), 0, 255);
                        rgb_b = math::clamp(math::cast<rgb_t>(255.0 * fn_xyz2rgb(var_b)), 0, 255);
                }

                //-------------------------------------------------------------------------------------------------

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

                //-------------------------------------------------------------------------------------------------

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

        //-------------------------------------------------------------------------------------------------

        void color::rgb2lab(rgb_t rgb_r, rgb_t rgb_g, rgb_t rgb_b,
                            scalar_t& cie_l, scalar_t& cie_a, scalar_t& cie_b)
        {
                scalar_t xyz_x, xyz_y, xyz_z;
                impl::rgb2xyz(rgb_r, rgb_g, rgb_b, xyz_x, xyz_y, xyz_z);
                impl::xyz2lab(xyz_x, xyz_y, xyz_z, cie_l, cie_a, cie_b);
        }

        //-------------------------------------------------------------------------------------------------

        void color::lab2rgb(scalar_t cie_l, scalar_t cie_a, scalar_t cie_b,
                            rgb_t& rgb_r, rgb_t& rgb_g, rgb_t& rgb_b)
        {
                scalar_t xyz_x, xyz_y, xyz_z;
                impl::lab2xyz(cie_l, cie_a, cie_b, xyz_x, xyz_y, xyz_z);
                impl::xyz2rgb(xyz_x, xyz_y, xyz_z, rgb_r, rgb_g, rgb_b);
        }

        //-------------------------------------------------------------------------------------------------

        image::image(size_t rows, size_t cols, const string_t& name)
                :       m_rgba(rows, cols),
                        m_name(name)
        {
        }

        //-------------------------------------------------------------------------------------------------

        image::image(const rgba_matrix_t& rgba, const string_t& name)
                :       m_rgba(rgba),
                        m_name(name)
        {
        }

        //-------------------------------------------------------------------------------------------------

        bool image::load(const rgba_matrix_t& rgba)
        {
                m_rgba = rgba;
                return false == empty();
        }

        //-------------------------------------------------------------------------------------------------

        bool image::load(const string_t& path)
        {
                QImage image;
                if (!image.load(path.c_str()))
                {
                        return false;
                }
                
                const int rows = image.height();
                const int cols = image.width();
                
                m_rgba.resize(rows, cols);                
                for (int r = 0; r < rows; r ++)
                {
                        for (int c = 0; c < cols; c ++)
                        {
                                const QRgb color = image.pixel(c, r);
                                m_rgba(r, c) = color::make_rgba(qRed(color), qGreen(color), qBlue(color), qAlpha(color));
                        }
                }
                
                return false == empty();
        }

        //-------------------------------------------------------------------------------------------------

        bool image::save(const string_t& path) const
        {
                QImage image(math::cast<int>(cols()),
                             math::cast<int>(rows()),
                             QImage::Format_RGB32);

                for (size_t r = 0; r < rows(); r ++)
                {
                        for (size_t c = 0; c < cols(); c ++)
                        {
                                const rgba_t color = m_rgba(r, c);
                                image.setPixel(c, r, qRgba(
                                        color::rgba2r(color),
                                        color::rgba2g(color),
                                        color::rgba2b(color),
                                        color::rgba2a(color)));
                        }
                }
                
                return image.save(path.c_str());
        }

        //-------------------------------------------------------------------------------------------------
}
