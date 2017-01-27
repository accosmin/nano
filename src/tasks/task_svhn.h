#pragma once

#include "task_mem_vision.h"

namespace nano
{
        class istream_t;
        class mat5_section_t;

        ///
        /// SVHN task:
        ///      - digit classification
        ///      - 32x32 color images as inputs
        ///      - 10 outputs (10 labels)
        ///
        /// http://ufldl.stanford.edu/housenumbers/
        ///
        class svhn_task_t final : public mem_vision_task_t
        {
        public:

                explicit svhn_task_t(const string_t& configuration = string_t());

        private:

                virtual bool populate() override;

                size_t load_binary(const string_t& path, const protocol);
                size_t load_pixels(const mat5_section_t&, const string_t&, const std::vector<int32_t>&, istream_t&);
                size_t load_labels(const mat5_section_t&, const string_t&, const std::vector<int32_t>&, size_t, const protocol, istream_t&);
        };
}

