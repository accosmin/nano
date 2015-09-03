#pragma once

#include "nanocv/task.h"

namespace ncv
{
        ///
        /// \brief synthetic task to detect digits
        ///
        /// parameters:
        ///     rows=32[16,128]         - patch size in pixels (rows)
        ///     cols=32[16,128]         - patch size in pixels (columns)
        ///     color=rgba[,luma]       - color mode
        ///     size=1024[256,64*1024]  - number of samples (training + validation)
        ///
        class NANOCV_PUBLIC synthetic_digits_task_t : public task_t
        {
        public:

                NANOCV_MAKE_CLONABLE(synthetic_digits_task_t,
                                     "synthetic task (digits classification), "\
                                     "parameters: rows=32[16,128],cols=32[16,128],"\
                                     "color=rgba[,luma],size=1024[256,64*1024]")

                // constructor
                explicit synthetic_digits_task_t(const string_t& configuration = string_t());

                // constructor
                synthetic_digits_task_t(size_t rows, size_t cols, color_mode, size_t size);

                // load images from the given directory
                virtual bool load(const string_t&) override;

                // access functions
                virtual size_t irows() const override { return m_rows; }
                virtual size_t icols() const override { return m_cols; }
                virtual size_t osize() const override { return 10; }
                virtual size_t fsize() const override { return m_folds; }
                virtual color_mode color() const override { return m_color; }

        private:

                // attributes
                size_t          m_rows;
                size_t          m_cols;
                size_t          m_folds;
                color_mode      m_color;
                size_t          m_size;
        };
}