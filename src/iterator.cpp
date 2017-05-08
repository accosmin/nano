#include "iterator.h"

namespace nano
{
        iterator_t::iterator_t(
                const task_t& task, const fold_t& fold) :
                iterator_t(task, fold, task.size(fold))
        {
        }

        iterator_t::iterator_t(
                const task_t& task, const fold_t& fold, const size_t batch0, const scalar_t factor) :
                m_task(task), m_fold(fold), m_batch(0), m_factor(0), m_begin(0), m_end(0)
        {
                reset(batch0, factor);
        }

        void iterator_t::reset(const size_t batch0, const scalar_t factor)
        {
                m_batch = static_cast<scalar_t>(batch0);
                m_factor = factor;
                m_begin = 0;
                m_end = 0;

                assert(batch0 > 0);
                assert(factor >= scalar_t(1));

                next();
        }

        void iterator_t::next()
        {
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
                m_batch = m_batch * m_factor;
        }
}
