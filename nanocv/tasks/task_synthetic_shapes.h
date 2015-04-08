#pragma once

#include "../task.h"

namespace ncv
{
        ///
        /// \brief synthetic task to detect random bright geometric shapes overimposed on random images
        ///
        /// parameters:
        ///     rows=32[16,48]          - patch size in pixels (rows)
        ///     cols=32[16,48]          - patch size in pixels (columns)
        ///     dims=4[2,10]            - number of outputs (= shape index in an image)
        ///     color=rgba[,luma]       - color mode
        ///     size=1024[256,16*1024]  - number of samples (training + validation)
        ///
        class NANOCV_PUBLIC synthetic_shapes_task_t : public task_t
        {
        public:

                NANOCV_MAKE_CLONABLE(synthetic_shapes_task_t,
                                     "synthetic task (shape classification), "\
                                     "parameters: rows=32[16,48],cols=32[16,48],dims=4[2,10],"\
                                     "color=rgba[,luma],size=1024[256,16*1024]")

                // constructor
                explicit synthetic_shapes_task_t(const string_t& configuration = string_t());

                // load images from the given directory
                virtual bool load(const string_t&) override;

                // access functions
                virtual size_t irows() const override { return m_rows; }
                virtual size_t icols() const override { return m_cols; }
                virtual size_t osize() const override { return m_outputs; }
                virtual size_t fsize() const override { return m_folds; }
                virtual color_mode color() const override { return m_color; }

        private:

                // attributes
                size_t          m_rows;
                size_t          m_cols;
                size_t          m_outputs;
                size_t          m_folds;
                color_mode      m_color;
                size_t          m_size;
        };
}
