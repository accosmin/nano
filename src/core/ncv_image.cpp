#include "ncv_image.h"
#include <QImage>

namespace ncv
{
        namespace impl
        {
                // note: the RGB - XYZ - CIELab color transformations are taken from:
                //      --- http://www.easyrgb.com/ ---

                //-------------------------------------------------------------------------------------------------

                scalar_t fn_xyz2lab(scalar_t t)
                {
                        if (t > 0.008856)
                                return std::pow(t, 1.0 / 3.0);
                        else
                                return 7.787 * t + 16.0 / 116.0;
                }

                //-------------------------------------------------------------------------------------------------

                scalar_t fn_lab2xyz(scalar_t t)
                {
                        const scalar_t t3 = t * t * t;
                        if (t3 > 0.008856)
                                return t3;
                        else
                                return (t - 16.0 / 116.0) / 7.787;
                }

                //-------------------------------------------------------------------------------------------------

                scalar_t fn_rgb2xyz(scalar_t t)
                {
                        if (t > 0.04045)
                                return std::pow((t + 0.055 ) / 1.055, 2.4);
                        else
                                return t / 12.92;
                }

                //-------------------------------------------------------------------------------------------------

                scalar_t fn_xyz2rgb(scalar_t t)
                {
                        if (t > 0.0031308)
                                return 1.055 * std::pow(t, 1.0 / 2.4) - 0.055;
                        else
                                return 12.92 * t;
                }

                //-------------------------------------------------------------------------------------------------

                void rgb2xyz(rgb_t rgb_r, rgb_t rgb_g, rgb_t rgb_b,
                             scalar_t& xyz_x, scalar_t& xyz_y, scalar_t& xyz_z)
                {
                        static const scalar_t inv_term1 = 1.0 / 255.0;

                        const scalar_t var_r = fn_rgb2xyz(rgb_r * inv_term1) * 100.0;
                        const scalar_t var_g = fn_rgb2xyz(rgb_g * inv_term1) * 100.0;
                        const scalar_t var_b = fn_rgb2xyz(rgb_b * inv_term1) * 100.0;

                        xyz_x = var_r * 0.4124 + var_g * 0.3576 + var_b * 0.1805;
                        xyz_y = var_r * 0.2126 + var_g * 0.7152 + var_b * 0.0722;
                        xyz_z = var_r * 0.0193 + var_g * 0.1192 + var_b * 0.9505;
                }

                //-------------------------------------------------------------------------------------------------

                void xyz2rgb(scalar_t xyz_x, scalar_t xyz_y, scalar_t xyz_z,
                             rgb_t& rgb_r, rgb_t& rgb_g, rgb_t& rgb_b)
                {
                        static const scalar_t inv_term1 = 1.0 / 100.0;

                        const scalar_t var_x = xyz_x * inv_term1;
                        const scalar_t var_y = xyz_y * inv_term1;
                        const scalar_t var_z = xyz_z * inv_term1;

                        const scalar_t var_r = var_x *  3.2406 + var_y * -1.5372 + var_z * -0.4986;
                        const scalar_t var_g = var_x * -0.9689 + var_y *  1.8758 + var_z *  0.0415;
                        const scalar_t var_b = var_x *  0.0557 + var_y * -0.2040 + var_z *  1.0570;

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

                        static const scalar_t inv_x_n = 1.0 / x_n;
                        static const scalar_t inv_y_n = 1.0 / y_n;
                        static const scalar_t inv_z_n = 1.0 / z_n;

                        const scalar_t var_x = fn_xyz2lab(xyz_x * inv_x_n);
                        const scalar_t var_y = fn_xyz2lab(xyz_y * inv_y_n);
                        const scalar_t var_z = fn_xyz2lab(xyz_z * inv_z_n);

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

                        static const scalar_t inv_term1 = 1.0 / 116.0;
                        static const scalar_t inv_term2 = 1.0 / 500.0;
                        static const scalar_t inv_term3 = 1.0 / 200.0;

                        // CIELab to XYZ
                        const scalar_t var_y = (cie_l + 16.0) * inv_term1;
                        const scalar_t var_x = cie_a * inv_term2 + var_y;
                        const scalar_t var_z = var_y - cie_b * inv_term3;

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

        bool image::save(const string_t& path, channel ch) const
        {
                QImage image(math::cast<int>(cols()),
                             math::cast<int>(rows()),
                             QImage::Format_RGB32);

                switch (ch)
                {
                case channel::red:
                        for (size_t r = 0; r < rows(); r ++)
                        {
                                for (size_t c = 0; c < cols(); c ++)
                                {
                                        const rgba_t color = m_rgba(r, c);
                                        image.setPixel(c, r, qRgba(color::rgba2r(color), 255, 255, 255));
                                }
                        }
                        break;

                case channel::green:
                        for (size_t r = 0; r < rows(); r ++)
                        {
                                for (size_t c = 0; c < cols(); c ++)
                                {
                                        const rgba_t color = m_rgba(r, c);
                                        image.setPixel(c, r, qRgba(255, color::rgba2g(color), 255, 255));
                                }
                        }
                        break;

                case channel::blue:
                        for (size_t r = 0; r < rows(); r ++)
                        {
                                for (size_t c = 0; c < cols(); c ++)
                                {
                                        const rgba_t color = m_rgba(r, c);
                                        image.setPixel(c, r, qRgba(255, 255, color::rgba2b(color), 255));
                                }
                        }
                        break;

                case channel::luma:
                        for (size_t r = 0; r < rows(); r ++)
                        {
                                for (size_t c = 0; c < cols(); c ++)
                                {
                                        const rgba_t color = m_rgba(r, c);
                                        const rgba_t luma = color::rgba2l(color);
                                        image.setPixel(c, r, qRgba(luma, luma, luma, 255));
                                }
                        }
                        break;
                }
                
                return image.save(path.c_str());
        }

        //-------------------------------------------------------------------------------------------------
}
