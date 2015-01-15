#pragma once

#include "task.h"

namespace ncv
{
        ///
        /// dummy task used for testing various components:
        ///     - fully resizable
        ///     - random image patches
        ///     - random training & test data
        ///     - random sample weights
        ///
        class dummy_task_t : public task_t
        {
        public:

                NANOCV_MAKE_CLONABLE(dummy_task_t, "test task")

                // constructor
                explicit dummy_task_t(const string_t& configuration = string_t());

                // change parameters
                void set_rows(size_t rows);
                void set_cols(size_t cols);
                void set_outputs(size_t outputs);
                void set_folds(size_t folds);
                void set_color(color_mode color);
                void set_size(size_t size);

                // setup samples
                void setup();

                // load images from the given directory
                virtual bool load(const string_t&) override { return true; }

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
