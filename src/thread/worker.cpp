#include "worker.h"
#include "tasks.h"

thread::pool_worker_t::pool_worker_t(tasks_t& queue)
        :       m_queue(queue)
{
}

void thread::pool_worker_t::operator()()
{
        while (true)
        {
                task_t task;

                // wait for a new task to be available in the queue
                {
                        std::unique_lock<std::mutex> lock(m_queue.m_mutex);

                        while ( !m_queue.m_stop &&
                                m_queue.m_tasks.empty())
                        {
                                m_queue.m_condition.wait(lock);
                        }

                        if (m_queue.m_stop)
                        {
                                m_queue.m_running = 0;
                                m_queue.m_tasks.clear();
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

                        m_queue.m_running --;
                        m_queue.m_condition.notify_all();
                }
        }
}
