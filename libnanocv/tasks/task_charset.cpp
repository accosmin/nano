#include "task_charset.h"
#include "libnanocv/class.h"
#include "libnanocv/vision/warp.h"
#include "libmath/gauss.hpp"
#include "libmath/clamp.hpp"
#include "libmath/random.hpp"
#include "libtext/to_string.hpp"
#include "libtext/from_params.hpp"
#include "libtensor/random.hpp"
#include "libnanocv/vision/convolve.hpp"
#include "libnanocv/vision/bilinear.hpp"
#include "libtensor/for_each.hpp"
#include "libtensor/transform.hpp"

#include "synth_bitstream_vera_sans_mono_bold.h"
#include "synth_bitstream_vera_sans_mono.h"
#include "synth_dejavu_sans_mono_bold.h"
#include "synth_dejavu_sans_mono.h"
#include "synth_droid_sans_mono.h"
#include "synth_liberation_mono_bold.h"
#include "synth_liberation_mono.h"
#include "synth_nimbus_mono_bold.h"
#include "synth_nimbus_mono.h"
#include "synth_oxygen_mono.h"

namespace ncv
{
        namespace text
        {
                template <>
                inline std::map<charset, std::string> enum_string<charset>()
                {
                        return
                        {
                                { charset::numeric, "digit" },
                                { charset::lalphabet, "lalpha" },
                                { charset::ualphabet, "ualpha" },
                                { charset::alphabet, "alpha" },
                                { charset::alphanumeric, "alphanum" }
                        };
                }
        }

        charset_task_t::charset_task_t(const string_t& configuration)
                :       task_t(configuration),
                        m_charset(text::from_params<charset>(configuration, "type", charset::numeric)),
                        m_rows(math::clamp(text::from_params<size_t>(configuration, "rows", 32), 16, 128)),
                        m_cols(math::clamp(text::from_params<size_t>(configuration, "cols", 32), 16, 128)),
                        m_folds(1),
                        m_color(text::from_params<color_mode>(configuration, "color", color_mode::rgba)),
                        m_size(math::clamp(text::from_params<size_t>(configuration, "size", 1024), 16, 1024 * 1024))
        {
        }

        charset_task_t::charset_task_t(
                charset cs, size_t rows, size_t cols, color_mode color, size_t size)
                :       charset_task_t(
                        "type=" + text::to_string(cs) + "," +
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

                tensor_t make_random_rgba_image(const size_t rows, const size_t cols, const rgba_t back_color,
                        const scalar_t max_noise, const scalar_t sigma)
                {
                        const scalar_t ir = ncv::color::get_red(back_color) / 255.0;
                        const scalar_t ig = ncv::color::get_green(back_color) / 255.0;
                        const scalar_t ib = ncv::color::get_blue(back_color) / 255.0;

                        tensor_t image(4, rows, cols);

                        // noisy background
                        random_t<scalar_t> back_noise(-max_noise, +max_noise);
                        tensor::for_each(image.matrix(0), [&] (scalar_t& value) { value = ir + back_noise(); });
                        tensor::for_each(image.matrix(1), [&] (scalar_t& value) { value = ig + back_noise(); });
                        tensor::for_each(image.matrix(2), [&] (scalar_t& value) { value = ib + back_noise(); });
                        image.matrix(3).setConstant(1.0);

                        // smooth background
                        const gauss_kernel_t<scalar_t> back_gauss(sigma);
                        ncv::convolve(back_gauss, image.matrix(0));
                        ncv::convolve(back_gauss, image.matrix(1));
                        ncv::convolve(back_gauss, image.matrix(2));

                        return image;
                }

                tensor_t alpha_blend(const tensor_t& mask, const tensor_t& img1, const tensor_t& img2)
                {
                        const auto op = [] (const auto a, const auto v1, const auto v2)
                        {
                                return (1.0 - a) * v1 + a * v2;
                        };

                        tensor_t imgb(4, mask.rows(), mask.cols());
                        tensor::transform(mask.matrix(3), img1.matrix(0), img2.matrix(0), imgb.matrix(0), op);
                        tensor::transform(mask.matrix(3), img1.matrix(1), img2.matrix(1), imgb.matrix(1), op);
                        tensor::transform(mask.matrix(3), img1.matrix(2), img2.matrix(2), imgb.matrix(2), op);
                        imgb.matrix(3).setConstant(1.0);

                        return imgb;
                }
        }

        bool charset_task_t::load(const string_t &)
        {
                const string_t characters =
                        "0123456789" \
                        "abcdefghijklmnopqrstuvwxyz" \
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

                const size_t n_chars = characters.size();

                const rgba_matrix_t char_patches[] =
                {
                        ncv::get_synth_bitstream_vera_sans_mono_bold(),
                        ncv::get_synth_bitstream_vera_sans_mono(),
                        ncv::get_synth_dejavu_sans_mono_bold(),
                        ncv::get_synth_dejavu_sans_mono(),
                        ncv::get_synth_droid_sans_mono(),
                        ncv::get_synth_liberation_mono_bold(),
                        ncv::get_synth_liberation_mono(),
                        ncv::get_synth_nimbus_mono_bold(),
                        ncv::get_synth_nimbus_mono(),
                        ncv::get_synth_oxygen_mono()
                };

                const size_t n_fonts = sizeof(char_patches) / sizeof(rgba_matrix_t);

                random_t<size_t> rng_protocol(1, 10);
                random_t<size_t> rng_output(obegin(), oend() - 1);
                random_t<size_t> rng_font(1, n_fonts);
                random_t<scalar_t> rng_gauss(0.0, 2.0);

                clear_memory(0);

                for (size_t f = 0; f < fsize(); f ++)
                {
                        for (size_t i = 0; i < m_size; i ++)
                        {
                                // random protocol: train vs. test (90% training, 10% testing)
                                const protocol p = (rng_protocol() < 9) ? protocol::train : protocol::test;

                                // random output class: character
                                const size_t o = rng_output();

                                // image: original object patch
                                const tensor_t opatch = ncv::color::to_rgba_tensor(
                                        get_object_patch(char_patches[rng_font() - 1], o, n_chars, 0.0));

                                // image: resize to the input size
                                tensor_t mpatch(4, irows(), icols());
                                ncv::bilinear(opatch.matrix(0), mpatch.matrix(0));
                                ncv::bilinear(opatch.matrix(1), mpatch.matrix(1));
                                ncv::bilinear(opatch.matrix(2), mpatch.matrix(2));
                                ncv::bilinear(opatch.matrix(3), mpatch.matrix(3));

                                // image: random warping
                                mpatch = ncv::warp(mpatch, warp_params(field_type::random, 0.1, 4.0, 16.0, 2.0));

                                // image: background & foreground layer
                                const auto bcolor = ncv::color::make_random_rgba();
                                const auto fcolor = ncv::color::make_opposite_random_rgba(bcolor);

                                const auto bnoise = 0.1;
                                const auto fnoise = 0.1;

                                const auto bsigma = rng_gauss();
                                const auto fsigma = rng_gauss();

                                const auto bpatch = make_random_rgba_image(irows(), icols(), bcolor, bnoise, bsigma);
                                const auto fpatch = make_random_rgba_image(irows(), icols(), fcolor, fnoise, fsigma);

                                // image: alpha-blend the background & foreground layer
                                const tensor_t patch = alpha_blend(mpatch, bpatch, fpatch);

                                image_t image;
                                switch (color())
                                {
                                case color_mode::luma:  image.load_luma(color::from_luma_tensor(patch)); break;
                                case color_mode::rgba:  image.load_rgba(color::from_rgba_tensor(patch)); break;
                                }

                                // generate image
                                add_image(image);

                                // generate sample
                                sample_t sample(n_images() - 1, sample_region(0, 0));
                                sample.m_label = string_t("char") + characters[o];
                                sample.m_target = ncv::class_target(o - obegin(), osize());
                                sample.m_fold = {f, p};
                                add_sample(sample);
                        }
                }

                return true;
        }

        size_t charset_task_t::osize() const
        {
                return oend() - obegin();
        }

        size_t charset_task_t::obegin() const
        {
                switch (m_charset)
                {
                case charset::numeric:          return 0;
                case charset::lalphabet:        return 0 + 10;
                case charset::ualphabet:        return 0 + 10 + 26;
                case charset::alphabet:         return 10;
                case charset::alphanumeric:     return 0;
                default:                        assert(false); return 0;
                }
        }

        size_t charset_task_t::oend() const
        {
                switch (m_charset)
                {
                case charset::numeric:          return 10;
                case charset::lalphabet:        return 10 + 26;
                case charset::ualphabet:        return 10 + 26 + 26;
                case charset::alphabet:         return 10 + 26 + 26;
                case charset::alphanumeric:     return 10 + 26 + 26;
                default:                        assert(false); return 0;
                }
        }
}
