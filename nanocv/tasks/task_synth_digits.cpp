#include "task_synth_digits.h"
#include "nanocv/class.h"
#include "syn_digits_courier.h"
#include "nanocv/math/clamp.hpp"
#include "nanocv/math/random.hpp"
#include "nanocv/vision/bilinear.hpp"
#include "nanocv/vision/gaussian.hpp"

namespace ncv
{
        synthetic_digits_task_t::synthetic_digits_task_t(const string_t& configuration)
                :       task_t(configuration),
                        m_rows(math::clamp(text::from_params<size_t>(configuration, "rows", 32), 16, 128)),
                        m_cols(math::clamp(text::from_params<size_t>(configuration, "cols", 32), 16, 128)),
                        m_folds(1),
                        m_color(text::from_params<color_mode>(configuration, "color", color_mode::rgba)),
                        m_size(math::clamp(text::from_params<size_t>(configuration, "size", 1024), 256, 64 * 1024))
        {
        }

        synthetic_digits_task_t::synthetic_digits_task_t(
                size_t rows, size_t cols, color_mode color, size_t size)
                :       synthetic_digits_task_t(
                        "rows=" + text::to_string(rows) + "," +
                        "cols=" + text::to_string(cols) + "," +
                        "color=" + text::to_string(color) + "," +
                        "size=" + text::to_string(size))
        {
        }

        namespace
        {
                template
                <
                        typename tmatrix
                >
                tmatrix get_object_patch(const tmatrix& image,
                        const size_t object_index, const size_t objects, const scalar_t max_offset)
                {
                        random_t<scalar_t> rng(-max_offset, max_offset);

                        const auto icols = static_cast<int>(image.cols());
                        const auto irows = static_cast<int>(image.rows());

                        const auto dx = static_cast<scalar_t>(icols) / static_cast<scalar_t>(objects);

                        const auto ppx = math::clamp(math::cast<int>(dx * object_index + rng()), 0, icols - 1);
                        const auto ppw = math::clamp(math::cast<int>(dx + rng()), 0, icols - ppx);

                        const auto ppy = math::clamp(math::cast<int>(rng()), 0, irows - 1);
                        const auto pph = math::clamp(math::cast<int>(irows + rng()), 0, irows - ppy);

                        return image.block(ppy, ppx, pph, ppw);
                }
        }

        bool synthetic_digits_task_t::load(const string_t &)
        {
                clear_memory(0);

                random_t<size_t> rng_protocol(1, 10);
                random_t<size_t> rng_output(1, osize());
                random_t<scalar_t> rng_gauss(scalar_t(0.5), scalar_t(2.0));

                const auto digit_patches = ncv::get_digits_courier();

                for (size_t f = 0; f < fsize(); f ++)
                {
                        for (size_t i = 0; i < m_size; i ++)
                        {
                                // random protocol: train vs. test
                                const protocol p = (rng_protocol() < 9) ? protocol::train : protocol::test;

                                // random output class: digit
                                const size_t o = rng_output();

                                //
                                const auto patch1 = get_object_patch(digit_patches, o - 1, osize(), 1.0);
                                const auto patch2 = bilinear(color::to_tensor(patch1), irows(), icols());
                                const auto patch3 = gaussian(patch2, rng_gauss());

//                                image.random_noise(color_channel::rgba, -40.0, +40.0, rng_gauss());

                                image_t image;
                                image.load(patch3);

//                                const rgba_t back_color = color::make_random_rgba();
//                                const rgba_t shape_color = color::make_opposite_random_rgba(back_color);

//                                // generate images
//                                switch (o)
//                                {
//                                case 1:         break;
//                                case 2:         make_filled_rect(image, shape_color); break;
//                                case 3:         make_hollow_rect(image, back_color, shape_color); break;
//                                case 4:         make_filled_ellipse(image, shape_color); break;
//                                case 5:         make_hollow_ellipse(image, back_color, shape_color); break;
//                                case 6:         make_cross(image, shape_color); break;
//                                case 7:         make_filled_up_triangle(image, shape_color); break;
//                                case 8:         make_hollow_up_triangle(image, back_color, shape_color); break;
//                                case 9:         make_filled_down_triangle(image, shape_color); break;
//                                case 10:        make_hollow_down_triangle(image, back_color, shape_color); break;
//                                default:        break;
//                                }

                                add_image(image);

                                // generate sample
                                sample_t sample(n_images() - 1, sample_region(0, 0));
                                sample.m_label = "digit" + text::to_string(o - 1);
                                sample.m_target = ncv::class_target(o - 1, osize());
                                sample.m_fold = {f, p};
                                add_sample(sample);
                        }
                }

                return true;
        }
}
