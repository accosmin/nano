#pragma once

#include "cortex/task.h"

namespace zob
{
        ///
        /// \brief synthetic task to classify characters
        ///
        /// parameters:
        ///     dims            - number of outputs
        ///     rows            - sample size in pixels (rows)
        ///     cols            - sample size in pixels (columns)
        ///     color           - color mode
        ///     size            - number of samples (training + validation)
        ///
        class ZOB_PUBLIC random_task_t : public task_t
        {
        public:

                ZOB_MAKE_CLONABLE(random_task_t,
                                     "random task: dims=2[2,10],rows=32[8,128],cols=32[8,128],"\
                                     "color=rgba[,luma],size=1024[16,1024*1024]")

                // constructor
                explicit random_task_t(const string_t& configuration = string_t());

                // load images from the given directory
                virtual bool load(const string_t&) override;

                // access functions
                virtual tensor_size_t irows() const override { return m_rows; }
                virtual tensor_size_t icols() const override { return m_cols; }
                virtual tensor_size_t osize() const override { return m_dims; }
                virtual size_t fsize() const override { return m_folds; }
                virtual color_mode color() const override { return m_color; }

        private:

                // attributes
                tensor_size_t   m_rows;
                tensor_size_t   m_cols;
                tensor_size_t   m_dims;
                size_t          m_folds;
                color_mode      m_color;
                size_t          m_size;
        };
}
