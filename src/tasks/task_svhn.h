#pragma once

#include "task_mem_vision.h"

namespace nano
{
        class istream_t;
        struct mat5_section_t;

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

                svhn_task_t();
                bool populate() override;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

        private:

                using dims_t = std::vector<int32_t>;
                using protocols_t = std::vector<protocol>;

                bool load_binary(const string_t& path, const protocol, const size_t count);
                bool load_pixels(const mat5_section_t&, const string_t&, const dims_t&, const size_t count, istream_t&);
                bool load_labels(const mat5_section_t&, const string_t&, const dims_t&, const protocol, const size_t count, istream_t&);

                // attributes
                string_t                m_dir;          ///< directory where to load the task from
                size_t                  m_folds{10};    ///<
        };
}
