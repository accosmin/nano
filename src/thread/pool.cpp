#include "pool.h"
#include "thread.h"
#include <cassert>
#include <algorithm>

namespace nano
{
        static std::size_t n_active_workers(const std::vector<worker_t>& workers)
        {
                return  static_cast<std::size_t>(std::count_if(workers.begin(), workers.end(),
                        [] (const auto& worker) { return worker.active(); }));
        }
}

nano::pool_t::pool_t() :
        pool_t(nano::n_threads())
{
}

nano::pool_t::pool_t(std::size_t active_threads)
{
        const auto n_workers = static_cast<std::size_t>(nano::n_threads());
        active_threads = std::max(std::size_t(1), active_threads);

        for (size_t i = 0; i < n_workers; ++ i)
        {
                m_workers.emplace_back(m_queue, i < active_threads);
        }
        for (size_t i = 0; i < n_workers; ++ i)
        {
                m_threads.emplace_back(std::ref(m_workers[i]));
        }

        assert(active_threads == nano::n_active_workers(m_workers));
}

nano::pool_t::~pool_t()
{
        // stop & join
        {
                const std::lock_guard<std::mutex> lock(m_queue.m_mutex);

                m_queue.m_stop = true;
                m_queue.m_condition.notify_all();
        }

        for (auto& thread : m_threads)
        {
                thread.join();
        }
}

void nano::pool_t::wait()
{
        // wait for all jobs to be taken and the workers to finish
        std::unique_lock<std::mutex> lock(m_queue.m_mutex);

        assert(nano::n_active_workers(m_workers) > 0);

        m_queue.m_condition.wait(lock, [&] ()
        {
                return m_queue.m_jobs.empty() && m_queue.m_running == 0;
        });
}

void nano::pool_t::activate(std::size_t count)
{
        const std::lock_guard<std::mutex> lock(m_queue.m_mutex);

        count = std::max(std::size_t(1), std::min(count, n_workers()));

        std::size_t crt_count = nano::n_active_workers(m_workers);
        assert(crt_count > 0);
        for (auto& worker :  m_workers)
        {
                if (crt_count == count)
                {
                        break;
                }

                else if (crt_count > count)
                {
                        if (worker.deactivate())
                        {
                                -- crt_count;
                        }
                }

                else if (crt_count < count)
                {
                        if (worker.activate())
                        {
                                ++ crt_count;
                        }
                }
        }

        assert(count == nano::n_active_workers(m_workers));

        m_queue.m_condition.notify_all();
}

std::size_t nano::pool_t::n_workers() const
{
        return m_workers.size();
}

std::size_t nano::pool_t::n_active_workers() const
{
        const std::lock_guard<std::mutex> lock(m_queue.m_mutex);

        return nano::n_active_workers(m_workers);
}

std::size_t nano::pool_t::n_jobs() const
{
        const std::lock_guard<std::mutex> lock(m_queue.m_mutex);

        return m_queue.m_jobs.size();
}
