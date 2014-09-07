#ifndef NANOCV_TASK_H
#define NANOCV_TASK_H

#include "common/manager.hpp"
#include "image.h"
#include "sample.h"

namespace ncv
{
        ///
        /// \brief manage tasks (register new ones, query and clone them)
        ///
        class task_t;
        typedef manager_t<task_t>                       task_manager_t;
        typedef task_manager_t::robject_t               rtask_t;

        ///
        /// \brief display a short description of the samples
        ///
        void print(const string_t& header, const samples_t& samples);

        ///
        /// \brief generic computer vision task consisting of a set of (annotated) images
        /// and a protocol (training + testing).
        /// samples for training & testing models can be drawn from these image.
        ///
        class task_t : public clonable_t<task_t>
	{
        public:

                task_t(const string_t& description)
                        :       clonable_t<task_t>(string_t(), description)
                {
                }
                
                // destructor
                virtual ~task_t() {}

                // load images from the given directory
                virtual bool load(const string_t& dir) = 0;

                // sample size
                rect_t sample_size() const;

                // sample region at a particular offset
                rect_t sample_region(coord_t x, coord_t y) const;               

                // save the task images to file (by grouping sample patchs into (grows, gcols) grids)
                void save_as_images(
                        const fold_t&, const string_t& basepath, size_t grows, size_t gcols,
                        size_t border = 8, rgba_t bkcolor = color::make_rgba(225, 225, 0)) const;
                void save_as_images(
                        const samples_t&, const string_t& basepath, size_t grows, size_t gcols,
                        size_t border = 8, rgba_t bkcolor = color::make_rgba(225, 225, 0)) const;

                // distinct labels
                strings_t labels() const;

                // access functions
                virtual size_t n_rows() const = 0;
                virtual size_t n_cols() const = 0;
                virtual size_t n_outputs() const = 0;
                virtual size_t n_folds() const = 0;
                virtual color_mode color() const = 0;

                size_t n_images() const { return m_images.size(); }
                const image_t& image(size_t i) const { return m_images[i]; }

                const samples_t& samples() const { return m_samples; }

        protected:

                void clear_images(size_t capacity)
                {
                        m_images.clear();
                        m_images.reserve(capacity);
                }

                void clear_samples(size_t capacity)
                {
                        m_samples.clear();
                        m_samples.reserve(capacity);
                }

                void clear_memory(size_t capacity)
                {
                        clear_images(capacity);
                        clear_samples(capacity);
                }

        protected:

                // attributes
                images_t                m_images;
                samples_t               m_samples;
        };
}

#endif // NANOCV_TASK_H
