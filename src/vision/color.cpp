#include "color.h"
#include "math/random.hpp"
#include "math/numeric.hpp"
#include "text/to_string.hpp"
#include "tensor/transform.hpp"

namespace nano
{
        rgba_t color::make_random_rgba()
        {
                nano::random_t<rgba_t> rng;

                return make_rgba(rng() & 0xFF, rng() & 0xFF, rng() & 0xFF, 255);
        }

        rgba_t color::make_opposite_random_rgba(const rgba_t source)
        {
                const auto cr = static_cast<int>(0xFF - get_red(source));
                const auto cg = static_cast<int>(0xFF - get_green(source));
                const auto cb = static_cast<int>(0xFF - get_blue(source));

                nano::random_t<int> rng(-55, +55);

                return make_rgba(static_cast<rgba_t>(nano::clamp(cr + rng(), 0, 255)),
                                 static_cast<rgba_t>(nano::clamp(cg + rng(), 0, 255)),
                                 static_cast<rgba_t>(nano::clamp(cb + rng(), 0, 255)),
                                 255);
        }

        tensor3d_t color::to_luma_tensor(const luma_matrix_t& luma)
        {
                const scalar_t scale = 1.0 / 255.0;

                tensor3d_t data(1, luma.rows(), luma.cols());
                tensor::transform(luma, data.matrix(0), [=] (luma_t l) { return scale * l; });

                return data;
        }

        tensor3d_t color::to_rgb_tensor(const rgba_matrix_t& rgba)
        {
                const scalar_t scale = 1.0 / 255.0;

                tensor3d_t data(3, rgba.rows(), rgba.cols());
                tensor::transform(rgba, data.matrix(0), [=] (rgba_t c) { return scale * color::get_red(c); });
                tensor::transform(rgba, data.matrix(1), [=] (rgba_t c) { return scale * color::get_green(c); });
                tensor::transform(rgba, data.matrix(2), [=] (rgba_t c) { return scale * color::get_blue(c); });

                return data;
        }

        tensor3d_t color::to_rgba_tensor(const rgba_matrix_t& rgba)
        {
                const scalar_t scale = 1.0 / 255.0;

                tensor3d_t data(4, rgba.rows(), rgba.cols());
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
                        return nano::cast<luma_t>(nano::clamp(value, tinput(0), tinput(255)));
                }
        }

        luma_matrix_t color::from_luma_tensor(const tensor3d_t& data)
        {
                luma_matrix_t luma(data.rows(), data.cols());

                switch (data.size<0>())
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

        rgba_matrix_t color::from_rgb_tensor(const tensor3d_t& data)
        {
                rgba_matrix_t rgba(data.rows(), data.cols());

                switch (data.size<0>())
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

        rgba_matrix_t color::from_rgba_tensor(const tensor3d_t& data)
        {
                rgba_matrix_t rgba(data.rows(), data.cols());

                switch (data.size<0>())
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
                default:                        return 255.0;
                }
        }

        std::ostream& operator<<(std::ostream& os, color_mode mode)
        {
                return os << nano::to_string(mode);
        }

}
