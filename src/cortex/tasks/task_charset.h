#pragma once

#include "cortex/task.h"

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
        class NANO_PUBLIC charset_task_t : public task_t
        {
        public:

                NANO_MAKE_CLONABLE(charset_task_t,
                                     "synthetic character classification: type=digit[lalpha,ualpha,alpha,alphanum],"\
                                     "rows=32[16,128],cols=32[16,128],"\
                                     "color=rgba[,luma],size=1024[16,1024*1024]")

                // constructor
                explicit charset_task_t(const string_t& configuration = string_t());

                // constructor
                charset_task_t(charset, tensor_size_t rows, tensor_size_t cols, color_mode, size_t size);

                // load images from the given directory
                virtual bool load(const string_t&) override;

                // access functions
                virtual tensor_size_t irows() const override { return m_rows; }
                virtual tensor_size_t icols() const override { return m_cols; }
                virtual tensor_size_t osize() const override;
                virtual size_t fsize() const override { return m_folds; }
                virtual color_mode color() const override { return m_color; }

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
