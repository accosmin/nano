#pragma once

#include "task_mem_vision.h"

namespace nano
{
        class archive_stream_t;

        ///
        /// STL10 task:
        ///     - object classification
        ///     - 96x96 color images as inputs
        ///     - 10 outputs (10 labels)
        ///
        /// http://www.stanford.edu/~acoates/stl10/
        ///
        class stl10_task_t final : public mem_vision_task_t
        {
        public:

                explicit stl10_task_t(const string_t& configuration = string_t());

        private:

                virtual bool populate() override;

                bool load_ifile(const string_t&, archive_stream_t&, const bool unlabed, const size_t count);
                bool load_gfile(const string_t&, archive_stream_t&, const size_t count);
                bool load_folds(const string_t&, archive_stream_t&, const size_t, const size_t, const size_t);

        private:

                struct sample_t
                {
                        explicit sample_t(const size_t image = 0, const tensor_index_t label = 0) :
                                m_image(image), m_label(label) {}

                        size_t          m_image;        ///< image index
                        tensor_index_t  m_label;        ///< label index
                };

                // attributes
                std::vector<sample_t>   m_samples;
        };
}

