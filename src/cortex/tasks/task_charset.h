#pragma once

#include "task_mem_vision.h"

namespace nano
{
        enum class charset
        {
                numeric,        ///< 0-9
                lalphabet,      ///< a-z
                ualphabet,      ///< A-Z
                alphabet,       ///< a-zA-Z
                alphanumeric,   ///< A-Za-z0-9
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
                        "synthetic character classification: type=digit[lalpha,ualpha,alpha,alphanum],"\
                        "rows=32[16,128],cols=32[16,128],"\
                        "color=rgba[,luma],size=1024[16,1024*1024]")

                ///
                /// \brief constructor
                ///
                explicit charset_task_t(const string_t& configuration = string_t());

                ///
                /// \brief short name of this task
                ///
                virtual string_t name() const override { return "charset"; }

                ///
                /// \brief load the task from the given directory (if possible)
                ///
                virtual bool load(const string_t& dir = string_t()) override;

        private:

                tensor_size_t obegin() const;
                tensor_size_t oend() const;

        private:

                // attributes
                charset         m_charset;
                tensor_size_t   m_rows;
                tensor_size_t   m_cols;
                size_t          m_folds;
                color_mode      m_color;
                size_t          m_size;
        };
}
