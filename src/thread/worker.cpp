#include "worker.h"
#include "queue.h"
#include <cassert>

namespace
{
        bool toggle(bool& value, const bool flag)
        {
                const bool changed = value == !flag;
                value = flag;
                return changed;
        }
}

nano::worker_t::worker_t(worker_queue_t& queue, const bool active) :
        m_queue(queue),
        m_active(active)
{
}

void nano::worker_t::operator()() const
{
        while (true)
        {
                worker_task_t task;

                // wait for a new task to be available in the queue
                {
                        std::unique_lock<std::mutex> lock(m_queue.m_mutex);

                        m_queue.m_condition.wait(lock, [&]
                        {
                                return m_queue.m_stop || (active() && !m_queue.m_tasks.empty());
                        });

                        if (m_queue.m_stop)
                        {
                                m_queue.m_tasks.clear();
                                m_queue.m_condition.notify_all();
                                break;
                        }

                        task = std::move(m_queue.m_tasks.front());
                        m_queue.m_tasks.pop_front();
                }

                // execute the task
                task();
        }
}

bool nano::worker_t::activate()
{
        return toggle(m_active, true);
}

bool nano::worker_t::deactivate()
{
        return toggle(m_active, false);
}

bool nano::worker_t::active() const
{
        return m_active;
}
