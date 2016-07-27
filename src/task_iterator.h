#pragma once

#include "task.h"

namespace nano
{
        ///
        /// \brief minibatch iterator over a task.
        ///     by default it produces fixed-size minibatches,
        //      but it can generate geometrically increasing minibatches.
        ///
        class NANO_PUBLIC task_iterator_t
        {
        public:

                ///
                /// \brief constructor
                ///
                task_iterator_t(const task_t&, const fold_t&, const size_t batch0, const scalar_t factor = scalar_t(1));

                ///
                /// \brief reset configuration, keep the task
                ///
                void reset(const size_t batch0, const scalar_t factor = scalar_t(1));

                ///
                /// \brief advance to the next minibatch by wrapping the fold if the end is reached.
                ///
                void next();

                ///
                /// \brief retrieve the [begin, end) sample range
                ///
                size_t begin() const { return m_begin; }
                size_t end() const { return m_end; }

                const fold_t& fold() const { return m_fold; }

        private:

                // attributes
                const task_t&   m_task;
                fold_t          m_fold;
                scalar_t        m_batch;                ///< current batch size
                scalar_t        m_factor;               ///< geometrically increasing factor of the batch size
                size_t          m_begin, m_end;         ///< sample range [begin, end)
        };
}
