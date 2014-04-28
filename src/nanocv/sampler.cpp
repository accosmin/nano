#include "sampler.h"

namespace ncv
{
//        /////////////////////////////////////////////////////////////////////////////////////////

//        samples_t prune_annotated(const task_t& task, const samples_t& samples)
//        {
//                samples_t pruned_samples;

//                // keep only the samples having targets associated
//                for (const sample_t& sample : samples)
//                {
//                        const image_t& image = task.image(sample.m_index);
//                        const vector_t target = image.make_target(sample.m_region);
//                        if (image.has_target(target))
//                        {
//                                pruned_samples.push_back(sample);
//                        }
//                }

//                return pruned_samples;
//        }

//        /////////////////////////////////////////////////////////////////////////////////////////

//        rect_t task_t::sample_size() const
//        {
//                return geom::make_size(n_cols(), n_rows());
//        }

//        /////////////////////////////////////////////////////////////////////////////////////////

//        rect_t task_t::sample_region(coord_t x, coord_t y) const
//        {
//                return geom::make_rect(x, y, n_cols(), n_rows());
//        }

//        /////////////////////////////////////////////////////////////////////////////////////////

//        samples_t task_t::make_samples(size_t istart, size_t icount, const rect_t& region)
//        {
//                samples_t samples(icount);
//                for (size_t i = 0; i < icount; i ++)
//                {
//                        samples[i].m_index = istart + i;
//                        samples[i].m_region = region;
//                }

//                return samples;
//        }

//        /////////////////////////////////////////////////////////////////////////////////////////

//        strings_t task_t::labels() const
//        {
//                // distinct labels
//                std::set<string_t> slabels;
//                for (const image_t& image : m_images)
//                {
//                        for (const annotation_t& annotation : image.m_annotations)
//                        {
//                                slabels.insert(annotation.m_label);
//                        }
//                }

//                strings_t result;
//                std::copy(slabels.begin(), slabels.end(), std::back_inserter(result));
//                return result;
//        }

//        /////////////////////////////////////////////////////////////////////////////////////////

//        void task_t::save_as_images(const fold_t& fold, const string_t& basepath, size_t grows, size_t gcols) const
//        {
//                const size_t border = 8;
//                const size_t rows = n_rows() * grows + border * (grows + 1);
//                const size_t cols = n_cols() * gcols + border * (gcols + 1);

//                const rgba_t back_color = color::make_rgba(225, 225, 0);

//                rgba_matrix_t rgba(rows, cols);

//                // process each label ...
//                const strings_t labels = this->labels();
//                for (size_t l = 0; l <= labels.size(); l ++)    // labels + non-annotated
//                {
//                        const string_t label = l < labels.size() ? labels[l] : string_t();

//                        // process all samples with this label ...
//                        const samples_t& samples = this->samples(fold);
//                        for (size_t i = 0, g = 1; i < samples.size(); g ++)
//                        {
//                                rgba.setConstant(back_color);

//                                // select samples
//                                samples_t gsamples;
//                                for ( ; i < samples.size() && gsamples.size() < grows * gcols; i ++)
//                                {
//                                        const sample_t& sample = samples[i];
//                                        const image_t& image = this->image(sample.m_index);
//                                        const rect_t& region = sample.m_region;

//                                        const string_t slabel = image.make_label(region);
//                                        if (slabel != label)
//                                        {
//                                                continue;
//                                        }

//                                        gsamples.push_back(sample);
//                                }

//                                // ... compose the image block
//                                for (size_t k = 0, r = 0; r < grows; r ++)
//                                {
//                                        for (size_t c = 0; c < gcols && k < gsamples.size(); c ++, k ++)
//                                        {
//                                                const sample_t& sample = gsamples[k];
//                                                const image_t& image = this->image(sample.m_index);
//                                                const rect_t& region = sample.m_region;

//                                                const size_t iy = n_rows() * r + border * (r + 1);
//                                                const size_t ix = n_cols() * c + border * (c + 1);
//                                                const size_t ih = n_rows();
//                                                const size_t iw = n_cols();

//                                                // image patch
//                                                rgba.block(iy, ix, ih, iw) = image.m_rgba.block(
//                                                        geom::top(region),
//                                                        geom::left(region),
//                                                        geom::height(region),
//                                                        geom::width(region));
//                                        }
//                                }

//                                // ... and save it
//                                const string_t path = basepath
//                                                + (label.empty() ? "" : ("_" + label))
//                                                + "_group" + text::to_string(g) + ".png";
//                                log_info() << "saving images to <" << path << "> ...";
//                                ncv::save_rgba(path, rgba);
//                        }
//                }
//        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
