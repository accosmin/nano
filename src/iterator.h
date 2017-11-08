#pragma once

#include "task.h"

namespace nano
{
        ///
        /// \brief iterate through a task's samples,
        ///     either using a fixed minibatch size if *factor* == 1
        ///     or using a geometrically increasing minibatch size if *factor* > 1.
        ///
        class iterator_t
        {
        public:
                ///
                /// \brief constructor
                ///
                iterator_t(const task_t& task, const fold_t& fold, const size_t batch0, const scalar_t factor = scalar_t(1)) :
                        m_task(task), m_fold(fold), m_batch(static_cast<scalar_t>(batch0)), m_factor(factor),
                        m_begin(0), m_end(0)
                {
                        next();
                }

                ///
                /// \brief change minibatch size
                ///
                void reset(const size_t batch0, const scalar_t factor = scalar_t(1))
                {
                        m_batch = static_cast<scalar_t>(batch0);
                        m_factor = factor;
                        m_begin = m_end = 0;
                        next();
                }

                ///
                /// \brief advance to the next minibatch by wrapping the fold if the end is reached
                /// NB: shuffles the task if the end is reached
                ///
                void next()
                {
                        assert(m_batch > 0);
                        assert(m_factor >= scalar_t(1));

                        const auto task_size = m_task.size(m_fold);
                        const auto batch_size = static_cast<size_t>(m_batch);

                        // wrap around the end
                        if (m_end + batch_size >= task_size)
                        {
                                m_task.shuffle(m_fold);
                                m_end = 0;
                        }

                        m_begin = m_end;
                        m_end = std::min(m_begin + batch_size, task_size);
                        if (batch_size >= task_size)
                        {
                                m_factor = 1;
                        }
                        m_batch *= m_factor;
                }

                ///
                /// \brief retrieve the [begin, end) sample range
                ///
                size_t begin() const { return m_begin; }
                size_t end() const { return m_end; }
                size_t size() const { return end() - begin(); }

                const task_t& task() const { return m_task; }
                const fold_t& fold() const { return m_fold; }

        private:

                // attributes
                const task_t&   m_task;         ///<
                const fold_t    m_fold;         ///<
                scalar_t        m_batch;        ///< current batch size
                scalar_t        m_factor;       ///< geometrically increasing factor of the batch size
                size_t          m_begin, m_end; ///< sample range [begin, end)
        };
}
