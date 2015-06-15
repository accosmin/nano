#pragma once

#include "sample.h"
#include "manager.hpp"
#include "vision/image.h"

namespace ncv
{
        class task_t;

        ///
        /// \brief manage tasks (register new ones, query and clone them)
        ///
        typedef manager_t<task_t>               task_manager_t;
        typedef task_manager_t::robject_t       rtask_t;

        NANOCV_PUBLIC task_manager_t& get_tasks();

        ///
        /// \brief describe the given samples
        ///
        NANOCV_PUBLIC void print(const string_t& header, const samples_t& samples);

        ///
        /// \brief generic computer vision task consisting of a set of (annotated) images
        /// and a protocol (training + testing).
        /// samples for training & testing models can be drawn from these image.
        ///
        class NANOCV_PUBLIC task_t : public clonable_t<task_t>
	{
        public:

                ///
                /// \brief constructor
                ///
                explicit task_t(const string_t& configuration)
                        : clonable_t<task_t>(configuration)
                {
                }
                
                // destructor
                virtual ~task_t() {}

                ///
                /// \brief load images from the given directory
                ///
                virtual bool load(const string_t& dir) = 0;

                ///
                /// \brief sample size (in pixels)
                ///
                rect_t sample_size() const;

                ///
                /// \brief sample region (in pixels) at a particular offset
                ///
                rect_t sample_region(coord_t x, coord_t y) const;               

                ///
                /// \brief save the task images to file (by grouping sample patchs into (grows, gcols) grids)
                ///
                void save_as_images(
                        const fold_t&, const string_t& basepath, size_t grows, size_t gcols,
                        size_t border = 8, rgba_t bkcolor = color::make_rgba(225, 225, 0)) const;

                ///
                /// \brief save the task images to file (by grouping sample patchs into (grows, gcols) grids)
                ///
                void save_as_images(
                        const samples_t&, const string_t& basepath, size_t grows, size_t gcols,
                        size_t border = 8, rgba_t bkcolor = color::make_rgba(225, 225, 0)) const;

                ///
                /// \brief describe the task
                ///
                void describe() const;

                ///
                /// \brief distinct labels
                ///
                strings_t labels() const;

                // access functions
                virtual size_t irows() const = 0;
                virtual size_t icols() const = 0;
                virtual size_t osize() const = 0;
                virtual size_t fsize() const = 0;
                virtual color_mode color() const = 0;

                size_t n_images() const { return m_images.size(); }
                const image_t& image(size_t i) const { return m_images[i]; }

                const samples_t& samples() const { return m_samples; }

        protected:

                ///
                /// \brief clear & reserve memory for images
                ///
                void clear_images(size_t capacity)
                {
                        m_images.clear();
                        m_images.reserve(capacity);
                }

                ///
                /// \brief clear & reserve memory for samples
                ///
                void clear_samples(size_t capacity)
                {
                        m_samples.clear();
                        m_samples.reserve(capacity);
                }

                ///
                /// \brief clear & reserve memory
                ///
                void clear_memory(size_t capacity)
                {
                        clear_images(capacity);
                        clear_samples(capacity);
                }

                ///
                /// \brief add a new image
                ///
                void add_image(const image_t& image);

                ///
                /// \brief add a new sample
                ///
                void add_sample(const sample_t& sample);

        private:

                // attributes
                images_t                m_images;       ///< input images (can be bigger than the samples)
                samples_t               m_samples;      ///< patch samples in images
        };
}
