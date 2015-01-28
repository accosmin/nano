#include "task_syn_dots.h"
#include "libnanocv/loss.h"
#include "libnanocv/util/random.hpp"

namespace ncv
{
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
                random_t<size_t> rng_output(1, osize());

                random_t<scalar_t> rng_gauss(scalar_t(1), math::cast<scalar_t>(icols() + irows()) / scalar_t(8));

                const coord_t border = 1;
                const coord_t minx = border;
                const coord_t maxx = math::cast<coord_t>(icols());
                const coord_t miny = border;
                const coord_t maxy = math::cast<coord_t>(irows());

                random_t<coord_t> rng_dotdx(coord_t(2), math::cast<coord_t>(icols() / 4));
                random_t<coord_t> rng_dotdy(coord_t(2), math::cast<coord_t>(irows() / 4));
                random_t<coord_t> rng_posx(minx, maxx);
                random_t<coord_t> rng_posy(miny, maxy);

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
                                        while (true)
                                        {
                                                const coord_t dx = rng_dotdx();
                                                const coord_t dy = rng_dotdy();
                                                const coord_t x = minx + (rng_posx() % (maxx - minx - dx));
                                                const coord_t y = miny + (rng_posy() % (maxy - miny - dy));

                                                const rect_t rect(x, y, dx, dy);

                                                // accept the dot only if it does not intersect with the previous ones!
                                                bool ok = true;
                                                for (size_t ioo = 0; ioo < io && ok; ioo ++)
                                                {
                                                        const rect_t& orect = dot_rects[ioo];

                                                        ok = (rect & rect_t(orect.left() - 1, orect.top() - 1,
                                                                            orect.width() + 1, orect.height() + 1)).empty();
                                                }

                                                if (ok)
                                                {
                                                        dot_rects.push_back(rect);
                                                        dot_rgbas.push_back(color::make_rgba(rng_red(), rng_green(), rng_blue()));
                                                        break;
                                                }
                                        }
                                }

                                // draw the dots
                                for (size_t io = 0; io < o; io ++)
                                {
                                        image.fill(dot_rects[io], dot_rgbas[io]);
                                }

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
