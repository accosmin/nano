#pragma once

#include "arch.h"
#include "protocol.h"
#include "manager.hpp"

namespace nano
{
        class task_t;

        ///
        /// \brief manage tasks (register new ones, query and clone them)
        ///
        using task_manager_t = manager_t<task_t>;
        using rtask_t = task_manager_t::trobject;

        NANO_PUBLIC task_manager_t& get_tasks();

        ///
        /// \brief machine learning task consisting of a collection of fixed-size 3D input tensors
        ///     split into training, validation and testing datasets.
        /// NB: the samples may be organized in folds depending on the established protocol.
        ///
        class NANO_PUBLIC task_t : public clonable_t<task_t>
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit task_t(const string_t& configuration = string_t()) :
                        clonable_t<task_t>(configuration)
                {
                }

                ///
                /// \brief destructor
                ///
                virtual ~task_t()
                {
                }

                ///
                /// \brief short name of this task
                ///
                virtual string_t name() const = 0;

                ///
                /// \brief load the task from the given directory (if possible)
                ///
                virtual bool load(const string_t& dir = string_t()) = 0;

                ///
                /// \brief print a short description
                ///
                void describe() const;

                ///
                /// \brief save the samples for the given fold as images (if possible) to the given path
                ///
                void save_as_images(const fold_t&, const string_t& basepath,
                        const tensor_size_t grows, const tensor_size_t gcols) const;

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
                /// \brief number of folds (not considering the protocol!)
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
        };

        ///
        /// \brief fixed-minibatch iterator over a task.
        ///
        class batch_iterator_t
        {
        public:

                batch_iterator_t(const task_t& task, const fold_t& fold, const size_t batch) :
                        m_task(task), m_fold(fold), m_batch(batch), m_begin(0), m_end(0)
                {
                        next();
                }

                void next()
                {
                        m_begin = m_end;
                        m_end = std::min(m_begin + m_batch, m_task.n_samples(m_fold));
                }

                void shuffle()
                {
                        m_task.shuffle(m_fold);
                        m_begin = m_end = 0;
                        next();
                }

                size_t begin() const { return m_begin; }
                size_t end() const { return m_end; }

        private:

                // attributes
                const task_t&   m_task;                 ///<
                fold_t          m_fold;
                const size_t    m_batch;
                size_t          m_begin, m_end;         ///< sample range [begin, end)
        };
}
