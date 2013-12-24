#include "task.h"

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
}
