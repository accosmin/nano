#pragma once

#include "arch.h"
#include "manager.h"
#include "protocol.h"

namespace nano
{
        ///
        /// \brief manage tasks (register new ones, query and clone them)
        ///
        class task_t;
        using task_manager_t = manager_t<task_t>;
        using rtask_t = task_manager_t::trobject;

        NANO_PUBLIC task_manager_t& get_tasks();

        ///
        /// \brief machine learning task consisting of a collection of fixed-size 3D input tensors
        ///     split into training, validation and testing datasets.
        /// NB: the samples may be organized in folds depending on the established protocol.
        ///
        class NANO_PUBLIC task_t : public clonable_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit task_t(const string_t& configuration = string_t()) : clonable_t(configuration) {}

                ///
                /// \brief populate task with samples
                ///
                virtual bool load() = 0;

                ///
                /// \brief input size
                ///
                virtual tensor_size_t idims() const = 0;
                virtual tensor_size_t irows() const = 0;
                virtual tensor_size_t icols() const = 0;

                ///
                /// \brief output size
                ///
                virtual tensor_size_t osize() const = 0;

                ///
                /// \brief number of folds
                ///
                virtual size_t n_folds() const = 0;

                ///
                /// \brief total number of samples
                ///
                virtual size_t n_samples() const = 0;

                ///
                /// \brief number of samples for the given fold
                ///
                virtual size_t n_samples(const fold_t&) const = 0;

                ///
                /// \brief randomly shuffle the samples associated for the given fold
                ///
                virtual void shuffle(const fold_t&) const = 0;

                ///
                /// \brief retrieve the 3D input tensor for a given sample
                ///
                virtual tensor3d_t input(const fold_t&, const size_t index) const = 0;

                ///
                /// \brief retrieve the target for a given sample
                ///
                virtual vector_t target(const fold_t&, const size_t index) const = 0;

                ///
                /// \brief retrieve the associated label (if any) for a given sample
                ///
                virtual string_t label(const fold_t&, const size_t index) const = 0;

                ///
                /// \brief retrieve the hash for a given sample
                ///
                virtual size_t hash(const fold_t&, const size_t index) const = 0;
        };
}
