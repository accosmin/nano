#include "task_charset.h"
#include "cortex/class.h"
#include "math/gauss.hpp"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "tensor/numeric.hpp"
#include "text/to_string.hpp"
#include "text/from_params.hpp"
#include "tensor/algorithm.hpp"
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
        static rgba_t make_random_rgba()
        {
                random_t<luma_t> rng(0, 255);
                return rgba_t{rng(), rng(), rng(), rng()};
        }

        static rgba_t make_random_opposite_rgba(const rgba_t& rgba)
        {
                random_t<int> rng(-55, +55);
                return  rgba_t
                {
                        static_cast<luma_t>(clamp(255 - static_cast<int>(rgba(0)) + rng(), 0, 255)),
                        static_cast<luma_t>(clamp(255 - static_cast<int>(rgba(1)) + rng(), 0, 255)),
                        static_cast<luma_t>(clamp(255 - static_cast<int>(rgba(2)) + rng(), 0, 255)),
                        static_cast<luma_t>(clamp(255 - static_cast<int>(rgba(3)) + rng(), 0, 255))
                };
        }

        template <>
        inline std::map<nano::charset, std::string> enum_string<nano::charset>()
        {
                return
                {
                        { nano::charset::digit,         "digit" },
                        { nano::charset::lalpha,        "lalpha" },
                        { nano::charset::ualpha,        "ualpha" },
                        { nano::charset::alpha,         "alpha" },
                        { nano::charset::alphanum,      "alphanum" }
                };
        }

        static tensor_size_t obegin(const charset cs)
        {
                switch (cs)
                {
                case charset::digit:            return 0;
                case charset::lalpha:           return 0 + 10;
                case charset::ualpha:           return 0 + 10 + 26;
                case charset::alpha:            return 10;
                case charset::alphanum:         return 0;
                default:                        assert(false); return 0;
                }
        }

        static tensor_size_t oend(const charset cs)
        {
                switch (cs)
                {
                case charset::digit:            return 10;
                case charset::lalpha:           return 10 + 26;
                case charset::ualpha:           return 10 + 26 + 26;
                case charset::alpha:            return 10 + 26 + 26;
                case charset::alphanum:         return 10 + 26 + 26;
                default:                        assert(false); return 0;
                }
        }

        static tensor_size_t osize(const charset cs)
        {
                return oend(cs) - obegin(cs);
        }

        template
        <
                typename tindex,
                typename tsize
        >
        tensor3d_t get_object_patch(const image_tensor_t& image,
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

                tensor3d_t ret(4, pph, ppw);
                ret.matrix(0) = image.matrix(0).block(ppy, ppx, pph, ppw).cast<scalar_t>() / 255.0;
                ret.matrix(1) = image.matrix(1).block(ppy, ppx, pph, ppw).cast<scalar_t>() / 255.0;
                ret.matrix(2) = image.matrix(2).block(ppy, ppx, pph, ppw).cast<scalar_t>() / 255.0;
                ret.matrix(3) = image.matrix(3).block(ppy, ppx, pph, ppw).cast<scalar_t>() / 255.0;

                return ret;
        }

        tensor3d_t make_random_rgba_image(const tensor_size_t rows, const tensor_size_t cols,
                const rgba_t back_color,
                const scalar_t max_noise, const scalar_t sigma)
        {
                // noisy background
                tensor3d_t image(4, rows, cols);
                image.matrix(0).setConstant(back_color(0) / 255.0);
                image.matrix(1).setConstant(back_color(1) / 255.0);
                image.matrix(2).setConstant(back_color(2) / 255.0);
                image.matrix(3).setConstant(1.0);

                tensor::add_random(nano::make_rng<scalar_t>(-max_noise, +max_noise),
                        image.matrix(0), image.matrix(1), image.matrix(2));

                // smooth background
                const nano::gauss_kernel_t<scalar_t> back_gauss(sigma);
                nano::convolve(back_gauss, image.matrix(0));
                nano::convolve(back_gauss, image.matrix(1));
                nano::convolve(back_gauss, image.matrix(2));

                return image;
        }

        tensor3d_t alpha_blend(const tensor3d_t& mask, const tensor3d_t& img1, const tensor3d_t& img2)
        {
                const auto op = [] (const auto a, const auto v1, const auto v2)
                {
                        return (1.0 - a) * v1 + a * v2;
                };

                tensor3d_t imgb(4, mask.rows(), mask.cols());
                tensor::transform(mask.matrix(3), img1.matrix(0), img2.matrix(0), imgb.matrix(0), op);
                tensor::transform(mask.matrix(3), img1.matrix(1), img2.matrix(1), imgb.matrix(1), op);
                tensor::transform(mask.matrix(3), img1.matrix(2), img2.matrix(2), imgb.matrix(2), op);
                imgb.matrix(3).setConstant(1.0);

                return imgb;
        }

        charset_task_t::charset_task_t(const string_t& configuration) : mem_vision_task_t(
                "charset",
                nano::from_params<color_mode>(configuration, "color", color_mode::rgb),
                nano::clamp(nano::from_params<tensor_size_t>(configuration, "irows", 32), 16, 128),
                nano::clamp(nano::from_params<tensor_size_t>(configuration, "icols", 32), 16, 128),
                nano::osize(nano::from_params<charset>(configuration, "type", charset::digit)),
                1),
                m_charset(nano::from_params<charset>(configuration, "type", charset::digit)),
                m_color(nano::from_params<color_mode>(configuration, "color", color_mode::rgb)),
                m_count(nano::clamp(nano::from_params<size_t>(configuration, "count", 1000), 100, 1024 * 1024))
        {
        }

        charset_task_t::charset_task_t(const charset type, const color_mode mode,
                const tensor_size_t irows, const tensor_size_t icols, const size_t count) :
                charset_task_t(
                "color=" + to_string(mode) + ",irows=" + to_string(irows) + ",icols=" + to_string(icols) +
                ",type=" + to_string(type) + ",count=" + to_string(count))
        {
        }

        bool charset_task_t::populate()
        {
                const string_t characters =
                        "0123456789" \
                        "abcdefghijklmnopqrstuvwxyz" \
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

                const size_t n_chars = characters.size();

                std::vector<image_tensor_t> char_patches;
                image_tensor_t char_patch;
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

                nano::random_t<tensor_size_t> rng_output(obegin(m_charset), oend(m_charset) - 1);
                nano::random_t<size_t> rng_font(1, n_fonts);
                nano::random_t<scalar_t> rng_gauss(0.0, 2.0);

                // generate samples
                for (size_t i = 0; i < m_count; ++ i)
                {
                        // random target: character
                        const tensor_index_t o = rng_output();

                        // image: original object patch
                        const tensor3d_t opatch = get_object_patch(char_patches[rng_font() - 1], o, n_chars, 0.0);

                        // image: resize to the input size
                        tensor3d_t mpatch(4, irows(), icols());
                        nano::bilinear(opatch.matrix(0), mpatch.matrix(0));
                        nano::bilinear(opatch.matrix(1), mpatch.matrix(1));
                        nano::bilinear(opatch.matrix(2), mpatch.matrix(2));
                        nano::bilinear(opatch.matrix(3), mpatch.matrix(3));

                        // image: random warping
                        mpatch = nano::warp(mpatch, warp_params_t(field_type::random, 0.1, 4.0, 16.0, 2.0));

                        // image: background & foreground layer
                        const auto bcolor = make_random_rgba();
                        const auto fcolor = make_random_opposite_rgba(bcolor);

                        const auto bnoise = 0.1;
                        const auto fnoise = 0.1;

                        const auto bsigma = rng_gauss();
                        const auto fsigma = rng_gauss();

                        const auto bpatch = make_random_rgba_image(irows(), icols(), bcolor, bnoise, bsigma);
                        const auto fpatch = make_random_rgba_image(irows(), icols(), fcolor, fnoise, fsigma);

                        // image: alpha-blend the background & foreground layer
                        const tensor3d_t patch = alpha_blend(mpatch, bpatch, fpatch);

                        image_t image;
                        image.from_tensor(patch);
                        switch (m_color)
                        {
                        case color_mode::luma:  image.make_luma(); break;
                        case color_mode::rgba:  image.make_rgba(); break;
                        case color_mode::rgb:   image.make_rgb(); break;
                        }

                        // generate image
                        add_chunk(image);

                        // generate sample
                        const auto fold = make_fold(0);
                        const auto target = class_target(o - nano::obegin(m_charset), nano::osize(m_charset));
                        const auto label = string_t("char") + characters[static_cast<size_t>(o)];
                        add_sample(fold, n_chunks() - 1, target, label);
                }

                return true;
        }
}
