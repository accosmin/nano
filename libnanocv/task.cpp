#include "task.h"
#include "sampler.h"
#include "libnanocv/logger.h"
#include "libnanocv/vision/image_grid.h"

namespace ncv
{
        task_manager_t& get_tasks()
        {
                return task_manager_t::instance();
        }

        void print(const string_t& header, const samples_t& samples)
        {
                const strings_t labels = ncv::labels(samples);

                for (const string_t& label : labels)
                {
                        sampler_t sampler(samples);
                        sampler.setup(sampler_t::stype::batch);
                        sampler.setup(label);

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
                return ncv::labels(m_samples);
        }

        void task_t::save_as_images(
                const fold_t& fold, const string_t& basepath, size_t grows, size_t gcols,
                size_t border, rgba_t bkcolor) const
        {
                // process each label ...
                const strings_t labels = this->labels();
                for (size_t l = 0; l < labels.size(); l ++)
                {
                        const string_t label = l < labels.size() ? labels[l] : string_t();

                        sampler_t sampler(*this);
                        sampler.setup(fold).setup(label);
                        const samples_t samples = sampler.get();

                        save_as_images(samples, basepath + (label.empty() ? "" : ("_" + label)),
                                       grows, gcols, border, bkcolor);
                }
        }

        void task_t::save_as_images(
                const samples_t& samples, const string_t& basepath, size_t grows, size_t gcols,
                size_t border, rgba_t bkcolor) const
        {
                for (size_t i = 0, g = 1; i < samples.size(); g ++)
                {
                        image_grid_t grid_image(irows(), icols(), grows, gcols, border, bkcolor);

                        // select samples
                        samples_t gsamples;
                        for ( ; i < samples.size() && gsamples.size() < grows * gcols; i ++)
                        {
                                gsamples.push_back(samples[i]);
                        }

                        // ... compose the image block
                        for (size_t k = 0, r = 0; r < grows; r ++)
                        {
                                for (size_t c = 0; c < gcols && k < gsamples.size(); c ++, k ++)
                                {
                                        const sample_t& sample = gsamples[k];

                                        grid_image.set(r, c, image(sample.m_index), sample.m_region);
                                }
                        }

                        // ... and save it
                        const string_t path = basepath + "_group" + text::to_string(g) + ".png";
                        log_info() << "saving images to <" << path << "> ...";
                        grid_image.image().save(path);
                }
        }

        void task_t::add_image(const image_t& image)
        {
                m_images.push_back(image);
        }

        void task_t::add_sample(const sample_t& sample)
        {
                // check sample to correspond to a valid region of a valid image
                if (    sample.m_index < n_images() &&
                        image(sample.m_index).valid(sample.m_region))
                {
                        m_samples.push_back(sample);
                }

                else
                {
                        log_error() << "task: cannot add invalid sample (index = "
                                    << sample.m_index << "/" << n_images() << ", region = "
                                    << sample.m_region << ")!";

                        throw std::runtime_error("invalid sample");
                }
        }

        void task_t::describe() const
        {
                log_info() << "images: " << n_images() << ".";
                log_info() << "sample: #rows = " << irows()
                           << ", #cols = " << icols()
                           << ", #outputs = " << osize()
                           << ", #folds = " << fsize() << ".";

                for (size_t f = 0; f < fsize(); f ++)
                {
                        for (protocol p : {protocol::train, protocol::test})
                        {
                                sampler_t sampler(*this);
                                sampler.setup({f, p});

                                ncv::print("fold [" + text::to_string(f + 1) + "/" + text::to_string(fsize()) + "] " +
                                           "protocol [" + text::to_string(p) + "]",
                                           sampler.all());
                        }
                }
        }
}
