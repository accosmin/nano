#include "charset.h"
#include "math/random.h"
#include "math/numeric.h"
#include "task_charset.h"
#include "tensor/numeric.h"
#include "tensor/algorithm.h"

#include "vision/warp.h"
#include "vision/gauss.h"
#include "vision/image_io.h"
#include "vision/convolve.h"
#include "vision/bilinear.h"

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

using namespace nano;

template <typename trng>
static rgba_t make_random_rgba(trng& rng)
{
        return rgba_t{rng(), rng(), rng(), rng()};
}

template <typename trng>
static rgba_t make_random_opposite_rgba(const rgba_t& rgba, trng& rng)
{
        return  rgba_t
        {
                static_cast<luma_t>(clamp(255 - static_cast<int>(rgba(0)) + rng(), 0, 255)),
                static_cast<luma_t>(clamp(255 - static_cast<int>(rgba(1)) + rng(), 0, 255)),
                static_cast<luma_t>(clamp(255 - static_cast<int>(rgba(2)) + rng(), 0, 255)),
                static_cast<luma_t>(clamp(255 - static_cast<int>(rgba(3)) + rng(), 0, 255))
        };
}

static tensor_size_t obegin(const charset_type cs)
{
        switch (cs)
        {
        case charset_type::digit:       return 0;
        case charset_type::lalpha:      return 0 + 10;
        case charset_type::ualpha:      return 0 + 10 + 26;
        case charset_type::alpha:       return 10;
        case charset_type::alphanum:    return 0;
        default:                        assert(false); return 0;
        }
}

static tensor_size_t oend(const charset_type cs)
{
        switch (cs)
        {
        case charset_type::digit:       return 10;
        case charset_type::lalpha:      return 10 + 26;
        case charset_type::ualpha:      return 10 + 26 + 26;
        case charset_type::alpha:       return 10 + 26 + 26;
        case charset_type::alphanum:    return 10 + 26 + 26;
        default:                        assert(false); return 0;
        }
}

static tensor_size_t osize(const charset_type cs)
{
        return oend(cs) - obegin(cs);
}

template <typename tindex, typename tsize, typename trng>
static void get_object_patch(const image_tensor_t& image,
        const tindex object_index, const tsize objects, trng& rng, tensor3d_t& patch)
{
        const auto icols = static_cast<int>(image.cols());
        const auto irows = static_cast<int>(image.rows());

        const auto dx = static_cast<scalar_t>(icols) / static_cast<scalar_t>(objects);

        const auto x = dx * static_cast<scalar_t>(object_index) + rng();

        const auto ppx = nano::clamp(std::lround(x), 0, icols - 1);
        const auto ppw = nano::clamp(std::lround(dx + rng()), 0, icols - ppx);

        const auto ppy = nano::clamp(std::lround(rng()), 0, irows - 1);
        const auto pph = nano::clamp(std::lround(static_cast<scalar_t>(irows) + rng()), 0, irows - ppy);

        patch.resize(4, pph, ppw);
        patch.matrix(0) = image.matrix(0).block(ppy, ppx, pph, ppw).template cast<scalar_t>() / scalar_t(255);
        patch.matrix(1) = image.matrix(1).block(ppy, ppx, pph, ppw).template cast<scalar_t>() / scalar_t(255);
        patch.matrix(2) = image.matrix(2).block(ppy, ppx, pph, ppw).template cast<scalar_t>() / scalar_t(255);
        patch.matrix(3) = image.matrix(3).block(ppy, ppx, pph, ppw).template cast<scalar_t>() / scalar_t(255);
}

template <typename trng>
static void make_random_rgba_image(const tensor_size_t rows, const tensor_size_t cols,
        const rgba_t back_color, const scalar_t sigma, trng& rng_noise, tensor3d_t& image)
{
        // noisy background
        image.resize(4, rows, cols);
        image.matrix(0).setConstant(back_color(0) / scalar_t(255));
        image.matrix(1).setConstant(back_color(1) / scalar_t(255));
        image.matrix(2).setConstant(back_color(2) / scalar_t(255));
        image.matrix(3).setConstant(1);

        nano::add_random(rng_noise, image.matrix(0), image.matrix(1), image.matrix(2));

        // smooth background
        const auto back_gauss = nano::make_gauss_kernel(sigma);
        nano::convolve(back_gauss, image.matrix(0));
        nano::convolve(back_gauss, image.matrix(1));
        nano::convolve(back_gauss, image.matrix(2));
}

static void alpha_blend(const tensor3d_t& mask, const tensor3d_t& img1, const tensor3d_t& img2, tensor3d_t& imgb)
{
        const auto op = [] (const auto& a, const auto& v1, const auto& v2, auto&& x)
        {
                x.array() = (1 - a.array()) * v1.array() + a.array() * v2.array();
        };

        imgb.resize(4, mask.rows(), mask.cols());
        op(mask.matrix(3), img1.matrix(0), img2.matrix(0), imgb.matrix(0));
        op(mask.matrix(3), img1.matrix(1), img2.matrix(1), imgb.matrix(1));
        op(mask.matrix(3), img1.matrix(2), img2.matrix(2), imgb.matrix(2));
        imgb.matrix(3).setConstant(1);
}

static string_t append_config(const string_t& params)
{
        return  to_params(params,
                "type", to_string(charset_type::digit) + "[" + concatenate(enum_values<charset_type>()) + "]",
                "color", "rgb[luma,rgba]",
                "irows", "32[12,256]", "icols", "32[12,256]", "count", "1000[32,1M]");
}

charset_task_t::charset_task_t(const string_t& params) : mem_vision_task_t(
        from_params<color_mode>(append_config(params), "color"),
        clamp(from_params<tensor_size_t>(append_config(params), "irows", 32), 12, 256),
        clamp(from_params<tensor_size_t>(append_config(params), "icols", 32), 12, 256),
        tensor3d_dims_t{osize(from_params<charset_type>(append_config(params), "type")), 1, 1},
        1, append_config(params))
{
}

bool charset_task_t::populate()
{
        const auto charset = from_params<charset_type>(config(), "type");
        const auto color = from_params<color_mode>(config(), "color");
        const auto count = clamp(from_params<size_t>(config(), "count"), 32, 1024 * 1024);

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

        auto rng_output = make_rng<tensor_size_t>(obegin(charset), oend(charset) - 1);
        auto rng_font = make_rng<size_t>(1, n_fonts);
        auto rng_gauss = make_rng<scalar_t>(0, 2);
        auto rng_rgba = make_rng<luma_t>(0, 255);
        auto rng_opposite_rgba = make_rng<int>(-55, +55);
        auto rng_offset = make_rng<scalar_t>(-1, +1);

        const auto bnoise = scalar_t(0.1);
        const auto fnoise = scalar_t(0.1);

        auto rng_bnoise = make_rng<scalar_t>(-bnoise, +bnoise);
        auto rng_fnoise = make_rng<scalar_t>(-fnoise, +fnoise);

        const auto irows = std::get<1>(idims());
        const auto icols = std::get<2>(idims());

        tensor3d_t opatch(4, irows, icols);
        tensor3d_t mpatch(4, irows, icols);
        tensor3d_t bpatch(4, irows, icols);
        tensor3d_t fpatch(4, irows, icols);

        // generate samples
        for (size_t i = 0; i < count; ++ i)
        {
                // random target: character
                const auto o = rng_output();

                // image: original object patch
                get_object_patch(char_patches[rng_font() - 1], o, n_chars, rng_offset, opatch);

                // image: resize to the input size
                mpatch.resize(4, irows, icols);
                nano::bilinear(opatch.matrix(0), mpatch.matrix(0));
                nano::bilinear(opatch.matrix(1), mpatch.matrix(1));
                nano::bilinear(opatch.matrix(2), mpatch.matrix(2));
                nano::bilinear(opatch.matrix(3), mpatch.matrix(3));

                // image: random warping
                const auto wnoise = scalar_t(0.1);
                const auto wsigma = scalar_t(4);
                const auto walpha = scalar_t(16);
                const auto wbeta = scalar_t(2);
                warp(mpatch, warp_type::mixed, wnoise, wsigma, walpha, wbeta);

                // image: background & foreground layer
                const auto bcolor = make_random_rgba(rng_rgba);
                const auto fcolor = make_random_opposite_rgba(bcolor, rng_opposite_rgba);

                const auto bsigma = rng_gauss();
                const auto fsigma = rng_gauss();

                make_random_rgba_image(irows, icols, bcolor, bsigma, rng_bnoise, bpatch);
                make_random_rgba_image(irows, icols, fcolor, fsigma, rng_fnoise, fpatch);

                // image: alpha-blend the background & foreground layer
                alpha_blend(mpatch, bpatch, fpatch, fpatch);

                image_t image;
                image.from_tensor(fpatch);
                switch (color)
                {
                case color_mode::luma:  image.make_luma(); break;
                case color_mode::rgba:  image.make_rgba(); break;
                case color_mode::rgb:   image.make_rgb(); break;
                }

                // generate image
                add_chunk(image, image.hash());

                // generate sample
                const auto fold = make_fold(0);
                const auto target = class_target(o - obegin(charset), osize(charset));
                const auto label = string_t("char") + characters[static_cast<size_t>(o)];
                add_sample(fold, n_chunks() - 1, target, label);
        }

        return true;
}
