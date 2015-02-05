#include "task_synthetic_shapes.h"
#include "libnanocv/loss.h"
#include "libnanocv/util/math.hpp"
#include "libnanocv/util/random.hpp"

namespace ncv
{
        synthetic_shapes_task_t::synthetic_shapes_task_t(const string_t& configuration)
                :       task_t(configuration),
                        m_rows(math::clamp(text::from_params<size_t>(configuration, "rows", 32), 16, 32)),
                        m_cols(math::clamp(text::from_params<size_t>(configuration, "cols", 32), 16, 32)),
                        m_outputs(math::clamp(text::from_params<size_t>(configuration, "dims", 4), 2, 16)),
                        m_folds(1),
                        m_color(text::from_params<color_mode>(configuration, "color", color_mode::rgba)),
                        m_size(math::clamp(text::from_params<size_t>(configuration, "size", 1024), 256, 16 * 1024))
        {
        }

        namespace
        {
                rgba_t make_transparent_color()
                {
                        return 0;
                }

                rgba_t make_light_color()
                {
                        random_t<rgba_t> rng_red(175, 255);
                        random_t<rgba_t> rng_green(175, 255);
                        random_t<rgba_t> rng_blue(175, 255);

                        return color::make_rgba(rng_red(), rng_green(), rng_blue());
                }

                rgba_t make_dark_color()
                {
                        random_t<rgba_t> rng_red(0, 125);
                        random_t<rgba_t> rng_green(0, 125);
                        random_t<rgba_t> rng_blue(0, 125);

                        return color::make_rgba(rng_red(), rng_green(), rng_blue());
                }

                rect_t make_rect(coord_t rows, coord_t cols)
                {
                        random_t<coord_t> rng(2, std::min(rows / 4, cols / 4));

                        const coord_t dx = rng();
                        const coord_t dy = rng();
                        const coord_t dw = rng();
                        const coord_t dh = rng();

                        return rect_t(dx, dy, cols - dx - dw, rows - dy - dh);
                }

                rect_t make_interior_rect(coord_t x, coord_t y, coord_t w, coord_t h)
                {
                        random_t<coord_t> rng(3, std::min(w / 4, h / 4));

                        const coord_t dx = rng();
                        const coord_t dy = rng();
                        const coord_t dw = rng();
                        const coord_t dh = rng();

                        return rect_t(x + dx, y + dy, w - dx - dw, h - dy - dh);
                }

                rect_t make_interior_rect(const rect_t& rect)
                {
                        return make_interior_rect(rect.left(), rect.top(), rect.width(), rect.height());
                }

                image_t make_filled_rect(coord_t rows, coord_t cols, rgba_t fill_color)
                {
                        const rect_t rect = make_rect(rows, cols);

                        image_t image(rows, cols, color_mode::rgba);

                        image.fill(make_transparent_color());
                        image.fill(rect, fill_color);

                        return image;
                }

                image_t make_hollow_rect(coord_t rows, coord_t cols, rgba_t fill_color)
                {
                        const rect_t rect = make_rect(rows, cols);

                        image_t image(rows, cols, color_mode::rgba);

                        image.fill(make_transparent_color());
                        image.fill(rect, fill_color);
                        image.fill(make_interior_rect(rect), make_transparent_color());

                        return image;
                }

                image_t make_filled_ellipse(coord_t rows, coord_t cols, rgba_t fill_color)
                {
                        const rect_t rect = make_rect(rows, cols);

                        image_t image(rows, cols, color_mode::rgba);

                        image.fill(make_transparent_color());
                        image.fill_ellipse(rect, fill_color);

                        return image;
                }

                image_t make_hollow_ellipse(coord_t rows, coord_t cols, rgba_t fill_color)
                {
                        const rect_t rect = make_rect(rows, cols);

                        image_t image(rows, cols, color_mode::rgba);

                        image.fill(make_transparent_color());
                        image.fill_ellipse(rect, fill_color);
                        image.fill_ellipse(make_interior_rect(rect), make_transparent_color());

                        return image;
                }
        }

        bool synthetic_shapes_task_t::load(const string_t &)
        {
                random_t<size_t> rng_protocol(1, 10);
                random_t<size_t> rng_output(1, osize());

                random_t<scalar_t> rng_gauss(scalar_t(1), math::cast<scalar_t>(icols() + irows()) / scalar_t(8));

                const coord_t rows = static_cast<coord_t>(irows());
                const coord_t cols = static_cast<coord_t>(icols());

                clear_memory(0);

                for (size_t f = 0; f < fsize(); f ++)
                {
                        for (size_t i = 0; i < m_size; i ++)
                        {
                                // random protocol: train vs. test
                                const protocol p = (rng_protocol() < 9) ? protocol::train : protocol::test;

                                // random output class: #dots
                                const size_t o = rng_output();

                                const bool is_dark_background = (rng_protocol() % 2) == 0;
                                const rgba_t back_color = is_dark_background ? make_dark_color() : make_light_color();
                                const rgba_t shape_color = is_dark_background ? make_light_color() : make_dark_color();

                                // generate random image background
                                image_t image(irows(), icols(), color());
                                image.fill(back_color);
                                image.random_noise(color_channel::rgba, -25.0, +25.0, rng_gauss());

                                // generate random shapes
                                image_t shape;

                                switch (o)
                                {
                                case 1:         shape = make_filled_rect(rows, cols, shape_color); break;
                                case 2:         shape = make_hollow_rect(rows, cols, shape_color); break;
                                case 3:         shape = make_filled_ellipse(rows, cols, shape_color); break;
                                case 4:         shape = make_hollow_ellipse(rows, cols, shape_color); break;
                                default:        break;
                                }

                                shape.random_noise(color_channel::rgba, -25.0, +25.0, rng_gauss() * 0.5);
                                image.alpha_blend(shape.rgba());

                                add_image(image);

                                // generate sample
                                sample_t sample(n_images() - 1, sample_region(0, 0));
                                switch (o)
                                {
                                case 1:         sample.m_label = "filled_rectangle"; break;
                                case 2:         sample.m_label = "hollow_rectangle"; break;
                                case 3:         sample.m_label = "filled_ellipse"; break;
                                case 4:         sample.m_label = "hollow_ellipse"; break;
                                default:        sample.m_label = "unkown"; break;
                                }

                                sample.m_target = ncv::class_target(o - 1, osize());
                                sample.m_fold = {f, p};
                                add_sample(sample);
                        }
                }

                return true;
        }
}
