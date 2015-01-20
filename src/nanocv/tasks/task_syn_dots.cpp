#include "task_syn_dots.h"
#include "loss.h"
#include "util/random.hpp"

namespace ncv
{
        namespace
        {
                scalars_t repell(const scalars_t& positions, scalar_t rows, scalar_t cols, size_t iters)
                {
                        const size_t particles = positions.size() / 2;

                        scalars_t new_positions(positions.size());
                        for (size_t p = 0; p < particles; p ++)
                        {
                                const scalar_t px = positions[2 * p + 0];
                                const scalar_t py = positions[2 * p + 1];

                                scalar_t dx = 0, dy = 0, dw = 0;
                                for (size_t q = 0; q < particles; q ++)
                                {
                                        if (p != q)
                                        {
                                                const scalar_t qx = positions[2 * q + 0];
                                                const scalar_t qy = positions[2 * q + 1];

                                                const scalar_t distance = math::square(px - qx) + math::square(py - qy);
                                                const scalar_t force = scalar_t(1) / (scalar_t(1e-6) + distance);

                                                dx += force * (px - qx);
                                                dy += force * (py - qy);
                                                dw += force;
                                        }
                                }

                                new_positions[2 * p + 0] = math::clamp(px + dx / dw, scalar_t(1), scalar_t(cols - 1));
                                new_positions[2 * p + 1] = math::clamp(py + dy / dw, scalar_t(1), scalar_t(cols - 1));
                        }

                        if (iters > 0)
                        {
                                return repell(new_positions, rows, cols, iters - 1);
                        }
                        else
                        {
                                return new_positions;
                        }
                }

                void repell(std::vector<rect_t>& rects, scalar_t rows, scalar_t cols, size_t iters)
                {
//                        scalars_t positions(rects.size() * 2);
//                        for (size_t i = 0; i < rects.size(); i ++)
//                        {
//                                positions[2 * i + 0] = rects[i].center().x();
//                                positions[2 * i + 1] = rects[i].center().y();
//                        }

////                        positions = repell(positions, rows, cols, iters);

//                        for (size_t i = 0; i < rects.size(); i ++)
//                        {
//                                const scalar_t x = positions[2 * i + 0];
//                                const scalar_t y = positions[2 * i + 1];

//                                rect_t& rect = rects[i];
//                                rect = rect_t(math::cast<coord_t>(x), math::cast<coord_t>(y), rect.width(), rect.height());
//                        }
                }
        }

        syn_dots_task_t::syn_dots_task_t(const string_t& configuration)
                :       task_t(configuration),
                        m_rows(math::clamp(text::from_params<size_t>(configuration, "rows", 16), 8, 32)),
                        m_cols(math::clamp(text::from_params<size_t>(configuration, "cols", 16), 8, 32)),
                        m_outputs(math::clamp(text::from_params<size_t>(configuration, "dims", 4), 2, 16)),
                        m_folds(1),
                        m_color(text::from_params<color_mode>(configuration, "color", color_mode::rgba)),
                        m_size(math::clamp(text::from_params<size_t>(configuration, "size", 1024), 256, 16 * 1024))
        {
        }

        bool syn_dots_task_t::load(const string_t &)
        {
                random_t<size_t> rng_protocol(1, 10);
                random_t<size_t> rng_output(1, n_outputs());

                random_t<scalar_t> rng_gauss(scalar_t(1), math::cast<scalar_t>(n_cols() + n_rows()) / scalar_t(8));

                const coord_t border = 1;
                const coord_t minx = border;
                const coord_t maxx = math::cast<coord_t>(n_cols());
                const coord_t miny = border;
                const coord_t maxy = math::cast<coord_t>(n_rows());

                random_t<coord_t> rng_dotdx(coord_t(2), math::cast<coord_t>(n_cols() / 4));
                random_t<coord_t> rng_dotdy(coord_t(2), math::cast<coord_t>(n_rows() / 4));
                random_t<coord_t> rng_posx(minx, maxx);
                random_t<coord_t> rng_posy(miny, maxy);

                random_t<rgba_t> rng_red(175, 255);
                random_t<rgba_t> rng_green(175, 255);
                random_t<rgba_t> rng_blue(175, 255);

                m_images.clear();
                m_samples.clear();

                for (size_t f = 0; f < n_folds(); f ++)
                {
                        for (size_t i = 0; i < m_size; i ++)
                        {
                                // random protocol: train vs. test
                                const protocol p = (rng_protocol() < 9) ? protocol::train : protocol::test;

                                // random output class: #dots
                                const size_t o = rng_output();

                                // generate sample
                                sample_t sample(m_images.size(), sample_region(0, 0));
                                sample.m_label = "count" + text::to_string(o);
                                sample.m_target = ncv::class_target(o - 1, n_outputs());
                                sample.m_fold = {f, p};
                                m_samples.push_back(sample);

                                // generate random image background
                                image_t image(n_rows(), n_cols(), color());
                                image.fill(color::make_rgba(rng_red(), rng_green(), rng_blue()));
                                image.random_noise(color_channel::rgba, -155.0, 55.0, rng_gauss());

//                                for (size_t io = 0; io < o; io ++)
//                                {
//                                        // generate random dot
//                                        const coord_t dx = rng_dotdx();
//                                        const coord_t dy = rng_dotdy();
//                                        const coord_t x = minx + (rng_posx() % (maxx - minx - dx));
//                                        const coord_t y = miny + (rng_posy() % (maxy - miny - dy));

//                                        image.fill(rect_t(x, y, dx, dy),
//                                                   color::make_rgba(rng_red(), rng_green(), rng_blue()));
//                                }

                                std::vector<rect_t> dot_rects;
                                std::vector<rgba_t> dot_rgbas;

                                // generate random dots
                                for (size_t io = 0; io < o; io ++)
                                {
                                        const coord_t dx = rng_dotdx();
                                        const coord_t dy = rng_dotdy();
                                        const coord_t x = minx + (rng_posx() % (maxx - minx - dx));
                                        const coord_t y = miny + (rng_posy() % (maxy - miny - dy));

                                        dot_rects.push_back(rect_t(x, y, dx, dy));
                                        dot_rgbas.push_back(color::make_rgba(rng_red(), rng_green(), rng_blue()));
                                }

                                // spread the dots across the image
                                const size_t iters = 32;
                                repell(dot_rects, math::cast<scalar_t>(n_rows()), math::cast<scalar_t>(n_cols()), iters);

                                // draw the dots
                                for (size_t io = 0; io < o; io ++)
                                {
                                        image.fill(dot_rects[io], dot_rgbas[io]);
                                }

                                m_images.push_back(image);
                        }
                }

                return true;
        }
}
