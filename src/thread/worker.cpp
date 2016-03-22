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

nano::worker_t::worker_t(queue_t& queue, const bool active) :
        m_queue(queue),
        m_active(active)
{
}

void nano::worker_t::operator()() const
{
        while (true)
        {
                job_t job;

                // wait for a new job to be available in the queue
                {
                        std::unique_lock<std::mutex> lock(m_queue.m_mutex);

                        m_queue.m_condition.wait(lock, [&]
                        {
                                return m_queue.m_stop || (active() && !m_queue.m_jobs.empty());
                        });

                        if (m_queue.m_stop)
                        {
                                m_queue.m_running = 0;
                                m_queue.m_jobs.clear();
                                m_queue.m_condition.notify_all();
                                break;
                        }

                        job = std::move(m_queue.m_jobs.front());
                        m_queue.m_jobs.pop_front();
                        m_queue.m_running ++;
                }

                // execute the job
                job();

                // announce that a job was completed
                {
                        const std::lock_guard<std::mutex> lock(m_queue.m_mutex);

                        assert(m_queue.m_running > 0);
                        m_queue.m_running --;
                        m_queue.m_condition.notify_all();
                }
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
