#pragma once

#include "task_mem_vision.h"

namespace nano
{
        enum class charset
        {
                digit,          ///< 0-9
                lalpha,         ///< a-z
                ualpha,         ///< A-Z
                alpha,          ///< a-zA-Z
                alphanum,       ///< A-Za-z0-9
        };

        ///
        /// \brief synthetic task to classify characters
        ///
        /// parameters:
        ///     type            - character set
        ///     rows            - sample size in pixels (rows)
        ///     cols            - sample size in pixels (columns)
        ///     color           - color mode
        ///     size            - number of samples (training + validation)
        ///
        class NANO_PUBLIC charset_task_t : public mem_vision_task_t
        {
        public:

                NANO_MAKE_CLONABLE(charset_task_t,
                        "synthetic character classification",
                        "type=digit[lalpha,ualpha,alpha,alphanum],"\
                        "color=rgb[,luma,rgba],irows=32[12,128],icols=32[12,128],count=1000[100,1M]")

                ///
                /// \brief constructor
                ///
                explicit charset_task_t(const string_t& configuration = string_t());

                ///
                /// \brief constructor
                ///
                charset_task_t(const charset, const color_mode,
                        const tensor_size_t irows, const tensor_size_t icols, const size_t count);

                ///
                /// \brief retrieve the color mdoe
                ///
                color_mode color() const { return m_color; }

        private:

                virtual bool populate() override;

        private:

                // attributes
                charset         m_charset;
                color_mode      m_color;
                size_t          m_count;
        };
}
