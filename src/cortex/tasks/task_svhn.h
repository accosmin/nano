#pragma once

#include "io/buffer.h"
#include "task_mem_vision.h"

namespace nano
{
        ///
        /// SVHN task:
        ///      - digit classification
        ///      - 32x32 color images as inputs
        ///      - 10 outputs (10 labels)
        ///
        /// http://ufldl.stanford.edu/housenumbers/
        ///
        class svhn_task_t : public mem_vision_task_t
        {
        public:

                NANO_MAKE_CLONABLE(svhn_task_t, "SVHN (object classification)")

                ///
                /// \brief constructor
                ///
                explicit svhn_task_t(const string_t& configuration = string_t());

                ///
                /// \brief short name of this task
                ///
                virtual string_t name() const override { return "svhn"; }

                ///
                /// \brief load the task from the given directory (if possible)
                ///
                virtual bool load(const string_t& dir = string_t()) override;

        private:

                // load binary file
                size_t load(const string_t& bfile, protocol p);

                // decode the uncompressed bytes (images + labels)
                size_t decode(const nano::buffer_t& image_data, const nano::buffer_t& label_data, const protocol p);
        };
}

