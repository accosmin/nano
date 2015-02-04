#include "task_synthetic_shapes.h"
#include "libnanocv/loss.h"
#include "libnanocv/util/random.hpp"

namespace ncv
{
        synthetic_shapes_task_t::synthetic_shapes_task_t(const string_t& configuration)
                :       task_t(configuration),
                        m_rows(math::clamp(text::from_params<size_t>(configuration, "rows", 16), 8, 32)),
                        m_cols(math::clamp(text::from_params<size_t>(configuration, "cols", 16), 8, 32)),
                        m_outputs(math::clamp(text::from_params<size_t>(configuration, "dims", 4), 2, 16)),
                        m_folds(1),
                        m_color(text::from_params<color_mode>(configuration, "color", color_mode::rgba)),
                        m_size(math::clamp(text::from_params<size_t>(configuration, "size", 1024), 256, 16 * 1024))
        {
        }

        namespace
        {
                rgba_matrix_t make_shape_rect(
                        coord_t rows, coord_t cols, coord_t posx, coord_t posy, coord_t sizex, coord_t sizey,
                        random_t<rgba_t>& rng_red, random_t<rgba_t>& rng_green, random_t<rgba_t>& rng_blue)
                {
                        rgba_matrix_t image(rows, cols);
                        image.setConstant(color::make_rgba(0, 0, 0, 0));

                        const rgba_t rgba = color::make_rgba(rng_red(), rng_green(), rng_blue());
                        image.block(posy, posx, sizey, sizex).setConstant(rgba);

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

                const coord_t border = 1;

                random_t<coord_t> rng_sizex(cols / 2, cols - 2 * border);
                random_t<coord_t> rng_sizey(rows / 2, rows - 2 * border);
                random_t<coord_t> rng_posx(0, cols);
                random_t<coord_t> rng_posy(0, rows);

                random_t<rgba_t> rng_red(175, 255);
                random_t<rgba_t> rng_green(175, 255);
                random_t<rgba_t> rng_blue(175, 255);

                clear_memory(0);

                for (size_t f = 0; f < fsize(); f ++)
                {
                        for (size_t i = 0; i < m_size; i ++)
                        {
                                // random protocol: train vs. test
                                const protocol p = (rng_protocol() < 9) ? protocol::train : protocol::test;

                                // random output class: #dots
                                const size_t o = rng_output();

                                // generate random image background
                                image_t image(irows(), icols(), color());
                                image.fill(color::make_rgba(rng_red(), rng_green(), rng_blue()));
                                image.random_noise(color_channel::rgba, -155.0, 55.0, rng_gauss());

                                // generate random shapes
                                const coord_t sizex = rng_sizex();
                                const coord_t sizey = rng_sizey();
                                const coord_t posx = border + rng_posx() % (cols - sizex - border);
                                const coord_t posy = border + rng_posy() & (rows - sizey - border);

                                // todo: generate other shapes
                                const rgba_matrix_t shape = make_shape_rect(rows, cols, posx, posy, sizex, sizey,
                                                                            rng_red, rng_green, rng_blue);
                                image.alpha_blend(shape);

                                add_image(image);

                                // generate sample
                                sample_t sample(n_images() - 1, sample_region(0, 0));
                                sample.m_label = "count" + text::to_string(o);
                                sample.m_target = ncv::class_target(o - 1, osize());
                                sample.m_fold = {f, p};
                                add_sample(sample);
                        }
                }

                return true;
        }
}
