#include "pool.h"
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

nano::thread_pool_t& nano::thread_pool_t::instance()
{
        static thread_pool_t the_pool;
        return the_pool;
}

nano::thread_pool_t::thread_pool_t()
{
        const auto n_workers = static_cast<std::size_t>(nano::logical_cpus());
        const auto n_active_threads = n_workers;

        m_workers.reserve(n_workers);
        for (size_t i = 0; i < n_workers; ++ i)
        {
                m_workers.emplace_back(m_queue, i < n_active_threads);
        }
        for (size_t i = 0; i < n_workers; ++ i)
        {
                m_threads.emplace_back(std::ref(m_workers[i]));
        }

        assert(n_active_threads == nano::n_active_workers(m_workers));
}

nano::thread_pool_t::~thread_pool_t()
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

void nano::thread_pool_t::activate(std::size_t count)
{
        const std::lock_guard<std::mutex> lock(m_queue.m_mutex);

        if (!count)
        {
                count = n_workers();
        }
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

std::size_t nano::thread_pool_t::n_workers() const
{
        return m_workers.size();
}

std::size_t nano::thread_pool_t::n_active_workers() const
{
        const std::lock_guard<std::mutex> lock(m_queue.m_mutex);

        return nano::n_active_workers(m_workers);
}

std::size_t nano::thread_pool_t::n_tasks() const
{
        const std::lock_guard<std::mutex> lock(m_queue.m_mutex);

        return m_queue.m_tasks.size();
}
