#pragma once

#include "task_mem_vision.h"

namespace nano
{
        ///
        /// STL10 task:
        ///      - object classification
        ///      - 96x96 color images as inputs
        ///     - 10 outputs (10 labels)
        ///
        /// http://www.stanford.edu/~acoates/stl10/
        ///
        class stl10_task_t : public mem_vision_task_t
        {
        public:

                NANO_MAKE_CLONABLE(stl10_task_t, "STL-10 (object classification)")

                ///
                /// \brief constructor
                ///
                explicit stl10_task_t(const string_t& configuration = string_t());

        private:

                virtual bool populate(const string_t& dir) override;

                // load binary files
                bool load_ifile(const string_t&, const char*, const size_t, const bool unlabed, const size_t count);
                bool load_gfile(const string_t&, const char*, const size_t, const size_t count);

                // build folds
                bool load_folds(const string_t&, const char*, const size_t, const size_t, const size_t, const size_t);
        };
}

