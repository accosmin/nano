#pragma once

#include "task_mem.hpp"
#include "vision/image.h"

namespace nano
{
        namespace detail
        {
                struct mem_vision_sample_t
                {
                        explicit mem_vision_sample_t(
                                const image_t& image = image_t(),
                                const rect_t& region = rect_t(),
                                const target_t& target = target_t()) :
                                m_image(image), m_region(region), m_target(target) {}

                        explicit mem_vision_sample_t(
                                const image_t& image,
                                const target_t& target) :
                                mem_vision_sample_t(image, rect_t(), target) {}

                        tensor3d_t input() const { return m_image.to_tensor(m_region); }
                        const target_t& target() const { return m_target; }

                        image_t         m_image;
                        rect_t          m_region;
                        target_t        m_target;
                };
        }

        ///
        /// \brief in-memory generic computer vision task consisting of images and
        ///     fixed-size rectangular samples from these images.
        ///
        class NANO_PUBLIC mem_vision_task_t : public mem_task_t<detail::mem_vision_sample_t>
        {
        public:

                ///
                /// \brief constructor
                ///
                mem_vision_task_t(
                        const string_t& name,
                        const tensor_size_t idims, const tensor_size_t irows, const tensor_size_t icols,
                        const tensor_size_t osize,
                        const size_t fsize) :
                        mem_task_t<detail::mem_vision_sample_t>(name, idims, irows, icols, osize, fsize) {}

                ///
                /// \brief destructor
                ///
                virtual ~mem_vision_task_t() {}
        };

        /*
                // attributes
                tensor_size_t           m_idims;        ///< input size
                tensor_size_t           m_icols;
                tensor_size_t           m_irows;
                tensor_size_t           m_osize;        ///< output size
                mutable storage_t       m_data;         ///< stored samples (training, validation, test)

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
                        const fold_t&, const string_t& basepath, coord_t grows, coord_t gcols,
                        coord_t border = 8, rgba_t bkcolor = color::make_rgba(225, 225, 0)) const;

                ///
                /// \brief save the task images to file (by grouping sample patchs into (grows, gcols) grids)
                ///
                void save_as_images(
                        const samples_t&, const string_t& basepath, coord_t grows, coord_t gcols,
                        coord_t border = 8, rgba_t bkcolor = color::make_rgba(225, 225, 0)) const;

                ///
                /// \brief describe the task
                ///
                void describe() const;

                ///
                /// \brief distinct labels
                ///
                strings_t labels() const;
        */
}
