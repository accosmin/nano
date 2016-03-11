#include "task_charset.h"
#include "cortex/class.h"
#include "math/gauss.hpp"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "tensor/random.hpp"
#include "text/to_string.hpp"
#include "tensor/for_each.hpp"
#include "text/from_params.hpp"
#include "tensor/transform.hpp"
#include "vision/warp.h"
#include "vision/image_io.h"
#include "vision/convolve.hpp"
#include "vision/bilinear.hpp"

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

namespace nano
{
        template <>
        inline std::map<nano::charset, std::string> enum_string<nano::charset>()
        {
                return
                {
                        { nano::charset::numeric, "digit" },
                        { nano::charset::lalphabet, "lalpha" },
                        { nano::charset::ualphabet, "ualpha" },
                        { nano::charset::alphabet, "alpha" },
                        { nano::charset::alphanumeric, "alphanum" }
                };
        }
}

namespace nano
{
        charset_task_t::charset_task_t(const string_t& configuration)
                :       task_t(configuration),
                        m_charset(nano::from_params<charset>(configuration, "type", charset::numeric)),
                        m_rows(nano::clamp(nano::from_params<tensor_size_t>(configuration, "rows", 32), 16, 128)),
                        m_cols(nano::clamp(nano::from_params<tensor_size_t>(configuration, "cols", 32), 16, 128)),
                        m_folds(1),
                        m_color(nano::from_params<color_mode>(configuration, "color", color_mode::rgba)),
                        m_size(nano::clamp(nano::from_params<size_t>(configuration, "size", 1024), 16, 1024 * 1024))
        {
        }

        charset_task_t::charset_task_t(
                charset cs, tensor_size_t rows, tensor_size_t cols, color_mode color, size_t size)
                :       charset_task_t(
                        "type=" + nano::to_string(cs) + "," +
                        "rows=" + nano::to_string(rows) + "," +
                        "cols=" + nano::to_string(cols) + "," +
                        "color=" + nano::to_string(color) + "," +
                        "size=" + nano::to_string(size))
        {
        }

        namespace
        {
                template
                <
                        typename tmatrix,
                        typename tindex,
                        typename tsize
                >
                tmatrix get_object_patch(const tmatrix& image,
                        const tindex object_index, const tsize objects, const scalar_t max_offset)
                {
                        nano::random_t<scalar_t> rng(-max_offset, max_offset);

                        const auto icols = static_cast<int>(image.cols());
                        const auto irows = static_cast<int>(image.rows());

                        const auto dx = static_cast<scalar_t>(icols) / static_cast<scalar_t>(objects);

                        const auto x = dx * static_cast<scalar_t>(object_index) + rng();

                        const auto ppx = nano::clamp(nano::cast<int>(x), 0, icols - 1);
                        const auto ppw = nano::clamp(nano::cast<int>(dx + rng()), 0, icols - ppx);

                        const auto ppy = nano::clamp(nano::cast<int>(rng()), 0, irows - 1);
                        const auto pph = nano::clamp(nano::cast<int>(irows + rng()), 0, irows - ppy);

                        return image.block(ppy, ppx, pph, ppw);
                }

                tensor_t make_random_rgba_image(const tensor_size_t rows, const tensor_size_t cols,
                        const rgba_t back_color,
                        const scalar_t max_noise, const scalar_t sigma)
                {
                        const scalar_t ir = nano::color::get_red(back_color) / 255.0;
                        const scalar_t ig = nano::color::get_green(back_color) / 255.0;
                        const scalar_t ib = nano::color::get_blue(back_color) / 255.0;

                        tensor_t image(4, rows, cols);

                        // noisy background
                        nano::random_t<scalar_t> back_noise(-max_noise, +max_noise);
                        tensor::for_each(image.matrix(0), [&] (scalar_t& value) { value = ir + back_noise(); });
                        tensor::for_each(image.matrix(1), [&] (scalar_t& value) { value = ig + back_noise(); });
                        tensor::for_each(image.matrix(2), [&] (scalar_t& value) { value = ib + back_noise(); });
                        image.matrix(3).setConstant(1.0);

                        // smooth background
                        const nano::gauss_kernel_t<scalar_t> back_gauss(sigma);
                        nano::convolve(back_gauss, image.matrix(0));
                        nano::convolve(back_gauss, image.matrix(1));
                        nano::convolve(back_gauss, image.matrix(2));

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

                std::vector<rgba_matrix_t> char_patches;
                rgba_matrix_t char_patch;
#define INSERT_IMAGE(name) \
                if (!nano::load_rgba_image(get_ ##name ##_name(), get_ ##name ##_data(), get_ ##name ##_size(), char_patch)) \
                { \
                        return false; \
                } \
                char_patches.push_back(char_patch);
                INSERT_IMAGE(synth_bitstream_vera_sans_mono_bold)
                INSERT_IMAGE(synth_bitstream_vera_sans_mono)
                INSERT_IMAGE(synth_dejavu_sans_mono_bold)
                INSERT_IMAGE(synth_dejavu_sans_mono)
                INSERT_IMAGE(synth_droid_sans_mono)
                INSERT_IMAGE(synth_liberation_mono_bold)
                INSERT_IMAGE(synth_liberation_mono)
                INSERT_IMAGE(synth_nimbus_mono_bold)
                INSERT_IMAGE(synth_nimbus_mono)
                INSERT_IMAGE(synth_oxygen_mono)
#undef INSERT_IMAGE

                const size_t n_fonts = char_patches.size();

                nano::random_t<size_t> rng_protocol(1, 10);
                nano::random_t<tensor_size_t> rng_output(obegin(), oend() - 1);
                nano::random_t<size_t> rng_font(1, n_fonts);
                nano::random_t<scalar_t> rng_gauss(0.0, 2.0);

                clear_memory(0);

                for (size_t f = 0; f < fsize(); ++ f)
                {
                        for (size_t i = 0; i < m_size; ++ i)
                        {
                                // random protocol: train vs. test (90% training, 10% testing)
                                const protocol p = (rng_protocol() < 9) ? protocol::train : protocol::test;

                                // random output class: character
                                const tensor_index_t o = rng_output();

                                // image: original object patch
                                const tensor_t opatch = nano::color::to_rgba_tensor(
                                        get_object_patch(char_patches[rng_font() - 1], o, n_chars, 0.0));

                                // image: resize to the input size
                                tensor_t mpatch(4, irows(), icols());
                                nano::bilinear(opatch.matrix(0), mpatch.matrix(0));
                                nano::bilinear(opatch.matrix(1), mpatch.matrix(1));
                                nano::bilinear(opatch.matrix(2), mpatch.matrix(2));
                                nano::bilinear(opatch.matrix(3), mpatch.matrix(3));

                                // image: random warping
                                mpatch = nano::warp(mpatch, warp_params(field_type::random, 0.1, 4.0, 16.0, 2.0));

                                // image: background & foreground layer
                                const auto bcolor = nano::color::make_random_rgba();
                                const auto fcolor = nano::color::make_opposite_random_rgba(bcolor);

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
                                sample.m_label = string_t("char") + characters[static_cast<size_t>(o)];
                                sample.m_target = nano::class_target(o - obegin(), osize());
                                sample.m_fold = {f, p};
                                add_sample(sample);
                        }
                }

                return true;
        }

        tensor_size_t charset_task_t::osize() const
        {
                return oend() - obegin();
        }

        tensor_size_t charset_task_t::obegin() const
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

        tensor_size_t charset_task_t::oend() const
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
