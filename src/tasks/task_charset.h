#pragma once

#include "charset.h"
#include "task_mem_vision.h"

namespace nano
{
        ///
        /// \brief synthetic task to classify characters
        ///
        /// parameters:
        ///     type    - character set
        ///     irows   - sample size in pixels (rows)
        ///     icols   - sample size in pixels (columns)
        ///     color   - color mode
        ///     count   - number of samples (training + validation)
        ///
        class charset_task_t final : public mem_vision_task_t
        {
        public:

                charset_task_t();
                bool populate() override;
                json_reader_t& config(json_reader_t&) final;
                json_writer_t& config(json_writer_t&) const final;

        private:

                // attributes
                charset_type    m_type{charset_type::digit};
                color_mode      m_color{color_mode::rgb};
                tensor_size_t   m_irows{32}, m_icols{32};
                size_t          m_count{1024};
        };
}
