#include "worker.h"
#include "tasks.h"
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

thread::worker_config_t::worker_config_t(const bool active)
        :       m_active(active)
{
}

bool thread::worker_config_t::activate()
{
        return toggle(m_active, true);
}

bool thread::worker_config_t::deactivate()
{
        return toggle(m_active, false);
}

bool thread::worker_config_t::active() const
{
        return m_active;
}

thread::worker_t::worker_t(tasks_t& queue, worker_config_t& config) :
        m_queue(queue),
        m_config(config)
{
}

void thread::worker_t::operator()()
{
        while (true)
        {
                task_t task;

                // wait for a new task to be available in the queue
                {
                        std::unique_lock<std::mutex> lock(m_queue.m_mutex);

                        m_queue.m_condition.wait(lock, [&]
                        {
                                return m_queue.m_stop || (m_config.active() && !m_queue.m_tasks.empty());
                        });

                        if (m_queue.m_stop)
                        {
                                m_queue.m_running = 0;
                                m_queue.m_tasks.clear();
                                lock.unlock();
                                m_queue.m_condition.notify_all();
                                break;
                        }

                        task = m_queue.m_tasks.front();
                        m_queue.m_tasks.pop_front();
                        m_queue.m_running ++;
                }

                // execute the task
                task();

                // announce that a task was completed
                {
                        const std::lock_guard<std::mutex> lock(m_queue.m_mutex);

                        assert(m_queue.m_running > 0);
                        m_queue.m_running --;
                }
                m_queue.m_condition.notify_all();
        }
}
