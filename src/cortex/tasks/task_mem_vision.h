#pragma once

#include "task_mem.hpp"
#include "vision/image.h"

namespace nano
{
        struct mem_vision_sample_t
        {
                explicit mem_vision_sample_t(
                        const size_t index = 0,
                        const rect_t& region = rect_t(),
                        const vector_t& target = vector_t(),
                        const string_t& label = string_t()) :
                        m_index(index), m_region(region), m_target(target), m_label(label) {}

                explicit mem_vision_sample_t(
                        const size_t index = 0,
                        const vector_t& target = vector_t(),
                        const string_t& label = string_t()) :
                        mem_vision_sample_t(index, rect_t(), target, label) {}

                auto index() const { return m_index; }
                auto input(const image_t& image) const { return m_region.empty() ? image.to_tensor() : image.to_tensor(m_region); }
                auto target() const { return m_target; }
                auto label() const { return m_label; }

                size_t          m_index;
                rect_t          m_region;
                vector_t        m_target;
                string_t        m_label;
        };

        ///
        /// \brief in-memory generic computer vision task consisting of images and
        ///     fixed-size rectangular samples from these images.
        ///
        class NANO_PUBLIC mem_vision_task_t : public mem_task_t<image_t, mem_vision_sample_t>
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
                        mem_task_t<image_t, mem_vision_sample_t>(name, idims, irows, icols, osize, fsize) {}

                ///
                /// \brief destructor
                ///
                virtual ~mem_vision_task_t() {}
        };

        /*
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
