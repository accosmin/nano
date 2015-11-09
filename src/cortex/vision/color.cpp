#include "color.h"
#include "math/abs.hpp"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "text/to_string.hpp"
#include "tensor/transform.hpp"

namespace cortex
{
        namespace
        {
                // note: the RGB - XYZ - CIELab color transformations are taken from:
                //      --- http://www.easyrgb.com/ ---                

                scalar_t fn_pow_12_5(scalar_t x)
                {
                        return std::pow(x, 2.4);
                }

                scalar_t fn_pow_5_12(scalar_t x)
                {
                        // 5/12 = 1/3 + 1/2 * 1/2 * 1/3;
                        const scalar_t cbx = std::cbrt(x);
                        return cbx * std::sqrt(std::sqrt(cbx));
                }

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

                class rgb2xyz_map
                {
                public:

                        // constructor
                        rgb2xyz_map()
                        {
                                for (rgba_t rgb = 0; rgb < 256; ++ rgb)
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

                void rgb2xyz(rgba_t rgb_r, rgba_t rgb_g, rgba_t rgb_b,
                             scalar_t& xyz_x, scalar_t& xyz_y, scalar_t& xyz_z)
                {
                        the_rgb2xyz_map(rgb_r, rgb_g, rgb_b, xyz_x, xyz_y, xyz_z);
                }

                void xyz2rgb(scalar_t xyz_x, scalar_t xyz_y, scalar_t xyz_z,
                             rgba_t& rgb_r, rgba_t& rgb_g, rgba_t& rgb_b)
                {
                        const scalar_t var_r = (xyz_x *  3.2406 + xyz_y * -1.5372 + xyz_z * -0.4986) / 100.0;
                        const scalar_t var_g = (xyz_x * -0.9689 + xyz_y *  1.8758 + xyz_z *  0.0415) / 100.0;
                        const scalar_t var_b = (xyz_x *  0.0557 + xyz_y * -0.2040 + xyz_z *  1.0570) / 100.0;

                        rgb_r = math::clamp(math::cast<rgba_t>(255.0 * fn_xyz2rgb(var_r)), rgba_t(0), rgba_t(255));
                        rgb_g = math::clamp(math::cast<rgba_t>(255.0 * fn_xyz2rgb(var_g)), rgba_t(0), rgba_t(255));
                        rgb_b = math::clamp(math::cast<rgba_t>(255.0 * fn_xyz2rgb(var_b)), rgba_t(0), rgba_t(255));
                }

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

        cielab_t color::make_cielab(rgba_t rgba)
        {
                cielab_t cielab;
                scalar_t x, y, z;

                rgb2xyz(get_red(rgba), get_green(rgba), get_blue(rgba), x, y, z);
                xyz2lab(x, y, z, cielab(0), cielab(1), cielab(2));
                cielab(3) = color::get_alpha(rgba);

                return cielab;
        }

        rgba_t color::make_rgba(const cielab_t& cielab)
        {
                rgba_t r, g, b;
                scalar_t x, y, z;

                lab2xyz(cielab(0), cielab(1), cielab(2), x, y, z);
                xyz2rgb(x, y, z, r, g, b);

                return make_rgba(r, g, b, static_cast<rgba_t>(cielab(3)));
        }

        rgba_t color::make_random_rgba()
        {
                math::random_t<rgba_t> rng;

                return make_rgba(rng() & 0xFF, rng() & 0xFF, rng() & 0xFF, 255);
        }

        rgba_t color::make_opposite_random_rgba(const rgba_t source)
        {
                const auto cr = static_cast<int>(0xFF - get_red(source));
                const auto cg = static_cast<int>(0xFF - get_green(source));
                const auto cb = static_cast<int>(0xFF - get_blue(source));

                math::random_t<int> rng(-55, +55);

                return make_rgba(static_cast<rgba_t>(math::clamp(cr + rng(), 0, 255)),
                                 static_cast<rgba_t>(math::clamp(cg + rng(), 0, 255)),
                                 static_cast<rgba_t>(math::clamp(cb + rng(), 0, 255)),
                                 255);
        }

        tensor_t color::to_luma_tensor(const luma_matrix_t& luma)
        {
                const scalar_t scale = 1.0 / 255.0;

                tensor_t data(1, luma.rows(), luma.cols());
                tensor::transform(luma, data.matrix(0), [=] (luma_t l) { return scale * l; });

                return data;
        }

        tensor_t color::to_rgb_tensor(const rgba_matrix_t& rgba)
        {
                const scalar_t scale = 1.0 / 255.0;

                tensor_t data(3, rgba.rows(), rgba.cols());
                tensor::transform(rgba, data.matrix(0), [=] (rgba_t c) { return scale * color::get_red(c); });
                tensor::transform(rgba, data.matrix(1), [=] (rgba_t c) { return scale * color::get_green(c); });
                tensor::transform(rgba, data.matrix(2), [=] (rgba_t c) { return scale * color::get_blue(c); });

                return data;
        }

        tensor_t color::to_rgba_tensor(const rgba_matrix_t& rgba)
        {
                const scalar_t scale = 1.0 / 255.0;

                tensor_t data(4, rgba.rows(), rgba.cols());
                tensor::transform(rgba, data.matrix(0), [=] (rgba_t c) { return scale * color::get_red(c); });
                tensor::transform(rgba, data.matrix(1), [=] (rgba_t c) { return scale * color::get_green(c); });
                tensor::transform(rgba, data.matrix(2), [=] (rgba_t c) { return scale * color::get_blue(c); });
                tensor::transform(rgba, data.matrix(3), [=] (rgba_t c) { return scale * color::get_alpha(c); });

                return data;
        }

        namespace
        {
                template
                <
                        typename tinput
                >
                luma_t to_byte(const tinput value)
                {
                        return math::cast<luma_t>(math::clamp(value, tinput(0), tinput(255)));
                }
        }

        luma_matrix_t color::from_luma_tensor(const tensor_t& data)
        {
                luma_matrix_t luma(data.rows(), data.cols());

                switch (data.dims())
                {
                case 1:
                        tensor::transform(data.matrix(0), luma,
                                          [=] (scalar_t l)
                        {
                                return to_byte(255.0 * l);
                        });
                        break;

                case 3:
                case 4:
                        tensor::transform(data.matrix(0), data.matrix(1), data.matrix(2), luma,
                                          [=] (scalar_t r, scalar_t g, scalar_t b)
                        {
                                return make_luma(to_byte(255.0 * r), to_byte(255.0 * g), to_byte(255.0 * b));
                        });
                        break;

                default:
                        throw std::runtime_error("can transform to luma only 1, 3 or 4-band tensors!");
                }

                return luma;
        }

        rgba_matrix_t color::from_rgb_tensor(const tensor_t& data)
        {
                rgba_matrix_t rgba(data.rows(), data.cols());

                switch (data.dims())
                {
                case 1:
                        tensor::transform(data.matrix(0), rgba, [=] (scalar_t l)
                        {
                                return make_rgba(to_byte(255.0 * l));
                        });
                        break;

                case 3:
                case 4:
                        tensor::transform(data.matrix(0), data.matrix(1), data.matrix(2), rgba,
                                          [=] (scalar_t r, scalar_t g, scalar_t b)
                        {
                                return make_rgba(to_byte(255.0 * r), to_byte(255.0 * g), to_byte(255.0 * b));
                        });
                        break;

                default:
                        throw std::runtime_error("can transform to rgb only 1, 3 or 4-band tensors!");
                }

                return rgba;
        }

        rgba_matrix_t color::from_rgba_tensor(const tensor_t& data)
        {
                rgba_matrix_t rgba(data.rows(), data.cols());

                switch (data.dims())
                {
                case 1:
                        tensor::transform(data.matrix(0), rgba, [=] (scalar_t l)
                        {
                                return make_rgba(to_byte(255.0 * l));
                        });
                        break;

                case 3:
                        tensor::transform(data.matrix(0), data.matrix(1), data.matrix(2), rgba,
                                          [=] (scalar_t r, scalar_t g, scalar_t b)
                        {
                                return make_rgba(to_byte(255.0 * r), to_byte(255.0 * g), to_byte(255.0 * b));
                        });
                        break;

                case 4:
                        tensor::transform(data.matrix(0), data.matrix(1), data.matrix(2), data.matrix(3), rgba,
                                          [=] (scalar_t r, scalar_t g, scalar_t b, scalar_t a)
                        {
                                return make_rgba(to_byte(255.0 * r), to_byte(255.0 * g), to_byte(255.0 * b), to_byte(255.0 * a));
                        });
                        break;

                default:
                        throw std::runtime_error("can transform to rgba only 1, 3 or 4-band tensors!");
                }

                return rgba;
        }

        scalar_t color::min(color_channel ch)
        {
                switch (ch)
                {
                case color_channel::red:        return 0.0;
                case color_channel::green:      return 0.0;
                case color_channel::blue:       return 0.0;
                case color_channel::luma:       return 0.0;
                case color_channel::cielab_l:   return 0.0;
                case color_channel::cielab_a:   return -86.1846;
                case color_channel::cielab_b:   return -107.864;
                default:                        return 0.0;
                }
        }

        scalar_t color::max(color_channel ch)
        {
                switch (ch)
                {
                case color_channel::red:        return 255.0;
                case color_channel::green:      return 255.0;
                case color_channel::blue:       return 255.0;
                case color_channel::luma:       return 255.0;
                case color_channel::cielab_l:   return 100.0;
                case color_channel::cielab_a:   return 98.2542;
                case color_channel::cielab_b:   return 94.4825;
                default:                        return 255.0;
                }
        }

        std::ostream& operator<<(std::ostream& os, color_mode mode)
        {
                return os << text::to_string(mode);
        }

}
