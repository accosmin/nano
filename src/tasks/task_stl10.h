#pragma once

#include "io/archive.h"
#include "task_mem_vision.h"

namespace nano
{
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

                stl10_task_t();
                bool populate() override;
                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;

        private:

                bool load_ifile(const string_t&, istream_t&, const bool unlabed, const size_t count);
                bool load_gfile(const string_t&, istream_t&, const size_t count);
                bool load_folds(const string_t&, istream_t&, const size_t, const size_t, const size_t);

                struct sample_t
                {
                        explicit sample_t(const size_t image = 0, const tensor_size_t label = 0) :
                                m_image(image), m_label(label) {}

                        size_t          m_image;        ///< image index
                        tensor_size_t   m_label;        ///< label index
                };

                // attributes
                string_t                m_dir;  ///< directory where to load the task from
                std::vector<sample_t>   m_samples;
        };
}
