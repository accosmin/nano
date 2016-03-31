#pragma once

#include "task_mem.hpp"
#include "vision/image.h"

namespace nano
{
        struct mem_vision_sample_t
        {
                explicit mem_vision_sample_t(
                        const size_t index = 0,
                        const vector_t& target = vector_t(),
                        const string_t& label = string_t(),
                        const rect_t& region = rect_t()) :
                        m_index(index), m_region(region), m_target(target), m_label(label)
                {
                }

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
                /// \brief constructor
                ///
                mem_vision_task_t(
                        const string_t& name,
                        const color_mode color, const tensor_size_t irows, const tensor_size_t icols,
                        const tensor_size_t osize,
                        const size_t fsize) :
                        mem_vision_task_t(name,
                        (color == color_mode::rgba ? 4 : (color == color_mode::rgb ? 3 : 1)),
                        irows, icols, osize, fsize) {}

                ///
                /// \brief destructor
                ///
                virtual ~mem_vision_task_t() {}

                ///
                /// \brief retrieve the number of images
                ///
                size_t n_images() const { return n_chunks(); }

                ///
                /// \brief retrieve the given image
                ///
                const image_t& image(const size_t index) const { return chunk(index); }
        };
}
