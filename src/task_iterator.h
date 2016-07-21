#pragma once

#include "task.h"

namespace nano
{
        enum class shuffle
        {
                on,
                off
        };

        ///
        /// \brief fixed-minibatch iterator over a task.
        ///
        template
        <
                shuffle tshuffle           ///< shuffle when reaching the end of the fold
        >
        class minibatch_iterator_t
        {
        public:

                ///
                /// \brief constructor
                ///
                minibatch_iterator_t(const task_t& task, const fold_t& fold, const size_t batch) :
                        m_task(task), m_fold(fold), m_batch(batch), m_begin(0), m_end(0)
                {
                        next();
                }

                ///
                /// \brief advance to the next minibatch by wrapping the fold if the end is reached.
                ///
                void next()
                {
                        const auto size = m_task.n_samples(m_fold);
                        if (m_end >= size)
                        {
                                switch (tshuffle)
                                {
                                case shuffle::on:       m_task.shuffle(m_fold); break;
                                default:                break;
                                }
                                m_end = 0;
                        }
                        m_begin = m_end;
                        m_end = std::min(m_begin + m_batch, size);

                }

                ///
                /// \brief retrieve the [begin, end) sample range
                ///
                size_t begin() const { return m_begin; }
                size_t end() const { return m_end; }

                const fold_t& fold() const { return m_fold; }

        private:

                // attributes
                const task_t&   m_task;                 ///<
                fold_t          m_fold;
                const size_t    m_batch;
                size_t          m_begin, m_end;         ///< sample range [begin, end)
        };
}
