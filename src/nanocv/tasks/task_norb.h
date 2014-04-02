#ifndef NANOCV_TASK_NORB_H
#define NANOCV_TASK_NORB_H

#include "task.h"

namespace ncv
{
        ///
        /// NORB task:
        ///      - 3D object recognition from shape
        ///      - 108x108 grayscale images as inputs (downscaled to 54x54 for memory reasons)
        ///      - 5 outputs (5 labels)
        ///
        /// http://www.cs.nyu.edu/~ylclab/data/norb-v1.0/
        ///
        class norb_task_t : public task_t
        {
        public:
                // constructor
                norb_task_t()
                        :       task_t("NORB (3D object recognition)")
                {
                }

                // create an object clone
                virtual rtask_t clone(const string_t&) const
                {
                        return rtask_t(new norb_task_t);
                }

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // access functions
                virtual size_t n_rows() const { return 54; }    // downscaled
                virtual size_t n_cols() const { return 54; }    // downscaled
                virtual size_t n_outputs() const { return 5; }
                virtual color_mode color() const { return color_mode::luma; }

        private:

                // load binary file
                size_t load(const string_t& bfile, protocol p);
                size_t load(const string_t& cfile, const string_t& dfile, protocol p);

                // build folds
                bool build_folds(size_t n_train, size_t n_test);
        };
}

#endif // NANOCV_TASK_NORB_H
