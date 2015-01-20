#pragma once

#include "task.h"

namespace ncv
{
        ///
        /// \brief synthetic task to count dots, represented as random bright rectangles, overimposed on random images
        ///
        /// parameters:
        ///     rows=16[8,32]           - patch size in pixels (rows)
        ///     cols=16[8,32]           - patch size in pixels (columns)
        ///     dims=4[2,16]            - number of outputs (= maximum number of dots in an image)
        ///     color=rgba[,luma]       - color mode
        ///     size=1024[256,16*1024]  - number of samples (training + validation)
        ///
        class syn_dots_task_t : public task_t
        {
        public:

                NANOCV_MAKE_CLONABLE(syn_dots_task_t,
                                     "synthetic test task to count dots, "\
                                     "rows=12[8,32],cols=12[8,32],dims=4[2,16],color=rgba[,luma],size=1024[256,16*1024]")

                // constructor
                explicit syn_dots_task_t(const string_t& configuration = string_t());

                // load images from the given directory
                virtual bool load(const string_t&) override;

                // access functions
                virtual size_t n_rows() const override { return m_rows; }
                virtual size_t n_cols() const override { return m_cols; }
                virtual size_t n_outputs() const override { return m_outputs; }
                virtual size_t n_folds() const override { return m_folds; }
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
