#pragma once

#include "libnanocv/task.h"

namespace ncv
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
        ///     type=digit[lalpha,ualpha,alpha,alphanum] - character set
        ///     rows=32[16,128]         - sample size in pixels (rows)
        ///     cols=32[16,128]         - sample size in pixels (columns)
        ///     color=rgba[,luma]       - color mode
        ///     size=1024[16,1024*1024] - number of samples (training + validation)
        ///
        class NANOCV_PUBLIC charset_task_t : public task_t
        {
        public:

                NANOCV_MAKE_CLONABLE(charset_task_t,
                                     "synthetic (character classification), "\
                                     "parameters: type=digit[lalpha,ualpha,alpha,alphanum],"\
                                     "rows=32[16,128],cols=32[16,128],"\
                                     "color=rgba[,luma],size=1024[16,1024*1024]")

                // constructor
                explicit charset_task_t(const string_t& configuration = string_t());

                // constructor
                charset_task_t(charset, size_t rows, size_t cols, color_mode, size_t size);

                // load images from the given directory
                virtual bool load(const string_t&) override;

                // access functions
                virtual size_t irows() const override { return m_rows; }
                virtual size_t icols() const override { return m_cols; }
                virtual size_t osize() const override;
                virtual size_t fsize() const override { return m_folds; }
                virtual color_mode color() const override { return m_color; }

        private:

                size_t obegin() const;
                size_t oend() const;

        private:

                // attributes
                charset         m_charset;
                size_t          m_rows;
                size_t          m_cols;
                size_t          m_folds;
                color_mode      m_color;
                size_t          m_size;
        };
}
