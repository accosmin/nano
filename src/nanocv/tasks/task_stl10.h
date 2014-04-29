#ifndef NANOCV_TASK_STL10_H
#define NANOCV_TASK_STL10_H

#include "task.h"

namespace ncv
{
        ///
        /// STL10 task:
        ///      - object classification
        ///      - 96x96 color images as inputs
        ///     - 10 outputs (10 labels)
        ///
        /// http://www.stanford.edu/~acoates/stl10/
        ///
        class stl10_task_t : public task_t
        {
        public:
                // constructor
                stl10_task_t()
                        :       task_t("STL-10 (object classification)")
                {
                }

                // create an object clone
                virtual rtask_t clone(const string_t&) const
                {
                        return rtask_t(new stl10_task_t);
                }

                // load images from the given directory
                virtual bool load(const string_t& dir);

                // access functions
                virtual size_t n_rows() const { return 96; }
                virtual size_t n_cols() const { return 96; }
                virtual size_t n_outputs() const { return 10; }
                virtual size_t n_folds() const { return 10; }
                virtual color_mode color() const { return color_mode::rgba; }
                                                   
        private:
                                                   
                // load binary files
                size_t load_binary(const string_t& ifile, const string_t& gfile);
                size_t load_binary(const string_t& ifile);

                // build folds
                bool load_folds(const string_t& ifile, size_t n_train, size_t n_unlabeled, size_t n_test);
        };
}

#endif // NANOCV_TASK_STL10_H
