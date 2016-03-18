#include "task.h"
#include "cortex/logger.h"
#include "text/to_string.hpp"
#include "vision/image_grid.h"

namespace nano
{
        /*
        void print(const string_t& header, const samples_t& samples)
        {
                const strings_t labels = nano::labels(samples);

                for (const string_t& label : labels)
                {
                        sampler_t sampler(samples);
                        sampler.push(label);

                        const samples_t lsamples = sampler.get();
                        log_info() << header << " [" << label
                                   << "]: count = " << lsamples.size()
                                   << "/" << samples.size() << ".";
                }
        }

        rect_t task_t::sample_size() const
        {
                return sample_region(0, 0);
        }

        rect_t task_t::sample_region(coord_t x, coord_t y) const
        {
                return rect_t(x, y, icols(), irows());
        }

        strings_t task_t::labels() const
        {
                return nano::labels(m_samples);
        }

        void task_t::save_as_images(
                const fold_t& fold, const string_t& basepath, coord_t grows, coord_t gcols,
                coord_t border, rgba_t bkcolor) const
        {
                // process each label ...
                const strings_t labels = this->labels();
                for (size_t l = 0; l < labels.size(); ++ l)
                {
                        const string_t label = l < labels.size() ? labels[l] : string_t();

                        sampler_t sampler(this->samples());
                        sampler.push(fold).push(label);
                        const samples_t samples = sampler.get();

                        save_as_images(samples, basepath + (label.empty() ? "" : ("_" + label)),
                                       grows, gcols, border, bkcolor);
                }
        }

        void task_t::save_as_images(
                const samples_t& samples, const string_t& basepath, coord_t grows, coord_t gcols,
                coord_t border, rgba_t bkcolor) const
        {
                for (size_t i = 0, g = 1; i < samples.size(); ++ g)
                {
                        image_grid_t grid_image(irows(), icols(), grows, gcols, border, bkcolor);

                        // compose the image block
                        for (coord_t r = 0; r < grows; ++ r)
                        {
                                for (coord_t c = 0; c < gcols && i < samples.size(); ++ c, ++ i)
                                {
                                        const sample_t& sample = samples[i];

                                        grid_image.set(r, c, image(sample.m_index), sample.m_region);
                                }
                        }

                        // ... and save it
                        const string_t path = basepath + "_group" + nano::to_string(g) + ".png";
                        log_info() << "saving images to <" << path << "> ...";
                        grid_image.image().save(path);
                }
        }

        void task_t::describe() const
        {
                log_info() << "images: " << n_images() << ".";
                log_info() << "sample: #rows = " << irows()
                           << ", #cols = " << icols()
                           << ", #outputs = " << osize()
                           << ", #folds = " << fsize() << ".";

                for (size_t f = 0; f < fsize(); ++ f)
                {
                        for (protocol p : {protocol::train, protocol::test})
                        {
                                sampler_t sampler(this->samples());
                                sampler.push({f, p});

                                nano::print("fold [" + nano::to_string(f + 1) + "/" + nano::to_string(fsize()) + "] " +
                                           "protocol [" + nano::to_string(p) + "]",
                                           sampler.get());
                        }
                }
        }
        */
}