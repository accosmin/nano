#include "task_synth_digits.h"
#include "nanocv/class.h"
#include "synth_digits.h"
#include "nanocv/math/gauss.hpp"
#include "nanocv/math/clamp.hpp"
#include "nanocv/math/random.hpp"
#include "nanocv/tensor/for_each.hpp"
#include "nanocv/tensor/transform.hpp"
#include "nanocv/vision/bilinear.hpp"
#include "nanocv/vision/separable_filter.hpp"

#include "nanocv/logger.h"

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
                        ncv::separable_filter(back_gauss, image.matrix(0));
                        ncv::separable_filter(back_gauss, image.matrix(1));
                        ncv::separable_filter(back_gauss, image.matrix(2));

                        return image;
                }

                tensor_t alpha_blend(const tensor_t& mask, const tensor_t& img1, const tensor_t& img2)
                {
                        const auto op = [] (const auto a, const auto v1, const auto v2)
                        {
                                return (1.0 - a) * v1 + a * v2;
                        };

                        tensor_t imgb(4, mask.rows(), mask.cols());
                        tensor::transform(mask.matrix(0), img1.matrix(0), img2.matrix(0), imgb.matrix(0), op);
                        tensor::transform(mask.matrix(1), img1.matrix(1), img2.matrix(1), imgb.matrix(1), op);
                        tensor::transform(mask.matrix(2), img1.matrix(2), img2.matrix(2), imgb.matrix(2), op);
                        imgb.matrix(3).setConstant(1.0);

                        return imgb;
                }
        }

        bool synthetic_digits_task_t::load(const string_t &)
        {
                clear_memory(0);

                random_t<size_t> rng_protocol(1, 10);
                random_t<size_t> rng_output(1, osize());
                random_t<scalar_t> rng_gauss(scalar_t(0.1), scalar_t(2.0));

                const auto digit_patches = ncv::get_synth_digits();

                for (size_t f = 0; f < fsize(); f ++)
                {
                        for (size_t i = 0; i < m_size; i ++)
                        {
                                // random protocol: train vs. test (90% training, 10% testing)
                                const protocol p = (rng_protocol() < 9) ? protocol::train : protocol::test;

                                // random output class: digit
                                const size_t o = rng_output();

                                // image: original object patch
                                const tensor_t opatch = ncv::color::to_rgba_tensor(
                                        get_object_patch(digit_patches, o - 1, osize(), 1.0));

                                log_info() << "opatch = " << opatch.dims() << "x" << opatch.rows() << "x" << opatch.cols();
                                log_info() << "opatch(0) = [" << opatch.matrix(0).minCoeff() << ", " << opatch.matrix(0).maxCoeff() << "]";
                                log_info() << "opatch(1) = [" << opatch.matrix(1).minCoeff() << ", " << opatch.matrix(1).maxCoeff() << "]";
                                log_info() << "opatch(2) = [" << opatch.matrix(2).minCoeff() << ", " << opatch.matrix(2).maxCoeff() << "]";
                                log_info() << "opatch(3) = [" << opatch.matrix(3).minCoeff() << ", " << opatch.matrix(3).maxCoeff() << "]";

                                // image: resize to the input size
                                tensor_t mpatch(4, irows(), icols());
                                ncv::bilinear(opatch.matrix(0), mpatch.matrix(0));
                                ncv::bilinear(opatch.matrix(1), mpatch.matrix(1));
                                ncv::bilinear(opatch.matrix(2), mpatch.matrix(2));
                                ncv::bilinear(opatch.matrix(3), mpatch.matrix(3));

                                log_info() << "mpatch(0) = [" << mpatch.matrix(0).minCoeff() << ", " << mpatch.matrix(0).maxCoeff() << "]";
                                log_info() << "mpatch(1) = [" << mpatch.matrix(1).minCoeff() << ", " << mpatch.matrix(1).maxCoeff() << "]";
                                log_info() << "mpatch(2) = [" << mpatch.matrix(2).minCoeff() << ", " << mpatch.matrix(2).maxCoeff() << "]";
                                log_info() << "mpatch(3) = [" << mpatch.matrix(3).minCoeff() << ", " << mpatch.matrix(3).maxCoeff() << "]";

                                // image: background & foreground layer
                                const auto bcolor = ncv::color::make_random_rgba();
                                const auto fcolor = ncv::color::make_opposite_random_rgba(bcolor);

                                const auto bnoise = 0.2;
                                const auto fnoise = 0.2;

                                const auto bsigma = rng_gauss();
                                const auto fsigma = 0.5 * rng_gauss();

                                const auto bpatch = make_random_rgba_image(irows(), icols(), bcolor, bnoise, bsigma);
                                const auto fpatch = make_random_rgba_image(irows(), icols(), fcolor, fnoise, fsigma);

                                // TODO: random warping (using Bottou's paper!)

                                // image: alpha-blend the background & foreground layer
                                const tensor_t patch = alpha_blend(mpatch, bpatch, fpatch);

                                image_t image;
                                switch (color())
                                {
                                case color_mode::luma:  image.load_luma(color::from_luma_tensor(patch)); break;
                                case color_mode::rgba:  image.load_rgba(color::from_rgb_tensor(patch)); break;
                                }

                                // generate image
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
