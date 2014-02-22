#include "task.h"
#include "text.h"
#include "common/logger.h"
#include "common/random.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        samples_t prune_annotated(const task_t& task, const samples_t& samples)
        {
                samples_t pruned_samples;

                // keep only the samples having targets associated
                for (const sample_t& sample : samples)
                {
                        const image_t& image = task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);
                        if (image.has_target(target))
                        {
                                pruned_samples.push_back(sample);
                        }
                }

                return pruned_samples;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        rect_t task_t::sample_size() const
        {
                return geom::make_size(n_cols(), n_rows());
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        rect_t task_t::sample_region(coord_t x, coord_t y) const
        {
                return geom::make_rect(x, y, n_cols(), n_rows());
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        samples_t task_t::make_samples(size_t istart, size_t icount, const rect_t& region)
        {
                samples_t samples(icount);
                for (size_t i = 0; i < icount; i ++)
                {
                        samples[i].m_index = istart + i;
                        samples[i].m_region = region;
                }

                return samples;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void task_t::save_as_images(const fold_t& fold, const string_t& basepath, size_t grows, size_t gcols) const
        {
                const size_t border = 16, radius = 4;
                const size_t rows = n_rows() * grows + border * (grows + 1);
                const size_t cols = n_cols() * gcols + border * (gcols + 1);

                const rgba_t back_color = color::make_rgba(225, 225, 0);

                random_t<rgba_t> rng(0, 255);

                rgbas_t label_colors(n_outputs());
                for (size_t o = 0; o < n_outputs(); o ++)
                {
                        label_colors[o] = color::make_rgba(rng(), rng(), rng());
                }

                rgba_matrix_t rgba(rows, cols);

                // process all samples ...
                const samples_t& samples = this->samples(fold);
                for (size_t i = 0, g = 1; i < samples.size(); i += grows * gcols, g ++)
                {
                        rgba.setConstant(back_color);

                        // ... compose the image block
                        for (size_t k = i, r = 0; r < grows; r ++)
                        {
                                for (size_t c = 0; c < gcols; c ++, k ++)
                                {
                                        if (k < samples.size())
                                        {
                                                const ncv::sample_t& sample = samples[k];
                                                const ncv::image_t& image = this->image(sample.m_index);
                                                const ncv::rect_t& region = sample.m_region;
                                                const ncv::vector_t target = image.make_target(sample.m_region);

                                                const size_t iy = n_rows() * r + border * (r + 1);
                                                const size_t ix = n_cols() * c + border * (c + 1);
                                                const size_t ih = n_rows();
                                                const size_t iw = n_cols();

                                                 // border ~ label/target
                                                if (image.has_target(target))
                                                {
                                                        for (size_t o = 0; o < n_outputs(); o ++)
                                                        {
                                                                if (target[o] > 0.0)
                                                                {
                                                                        const rgba_t color = label_colors[o];
                                                                        rgba.block(iy - radius,
                                                                                   ix - radius,
                                                                                   ih + radius * 2,
                                                                                   iw + radius * 2).setConstant(color);
                                                                        break;
                                                                }
                                                        }
                                                }

                                                // image patch
                                                rgba.block(iy, ix, ih, iw) = image.m_rgba.block(
                                                                        geom::top(region),
                                                                        geom::left(region),
                                                                        geom::height(region),
                                                                        geom::width(region));
                                        }
                                }
                        }

                        // ... and save it
                        const string_t path = basepath + "_group" + text::to_string(g) + ".png";
                        log_info() << "saving images to <" << path << "> ...";
                        ncv::save_rgba(path, rgba);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
