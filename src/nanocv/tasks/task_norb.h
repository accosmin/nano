#pragma once

#include "cortex/task.h"

namespace ncv
{
        ///
        /// NORB task:
        ///      - 3D object recognition from shape
        ///      - 108x108 grayscale images as inputs
        ///      - 5 outputs (5 labels)
        ///
        /// http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/
        ///
        class norb_task_t : public task_t
        {
        public:
                
                NANOCV_MAKE_CLONABLE(norb_task_t, "NORB (3D object recognition)")

                // constructor
                explicit norb_task_t(const string_t& configuration = string_t());

                // load images from the given directory
                virtual bool load(const string_t& dir) override;

                // access functions
                virtual size_t irows() const override { return 108; }
                virtual size_t icols() const override { return 108; }
                virtual size_t osize() const override { return 5; }
                virtual size_t fsize() const override { return 1; }
                virtual color_mode color() const override { return color_mode::luma; }

        private:

                // load binary file
                bool load(const string_t& bfile, protocol p, size_t count);
                bool load(const string_t& ifile, const string_t& gfile, protocol p, size_t count);
        };
}

