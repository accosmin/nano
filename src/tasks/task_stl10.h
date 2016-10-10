#pragma once

#include "io/buffer.h"
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
        class stl10_task_t : public mem_vision_task_t
        {
        public:

                NANO_MAKE_CLONABLE(stl10_task_t, "dir=.")

                ///
                /// \brief constructor
                ///
                explicit stl10_task_t(const string_t& configuration = string_t());

        private:

                virtual bool populate() override;

                // load binary files
                bool load_ifile(const string_t&, const buffer_t&, const bool unlabed, const size_t count);
                bool load_gfile(const string_t&, const buffer_t&, const size_t count);

                // build folds
                bool load_folds(const string_t&, const buffer_t&, const size_t, const size_t, const size_t);

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

