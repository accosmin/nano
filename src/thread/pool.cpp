#include "pool.h"
#include "thread.h"
#include <cassert>
#include <algorithm>

namespace thread
{
        static std::size_t n_active_workers(const std::vector<worker_t>& workers)
        {
                return  static_cast<std::size_t>(std::count_if(workers.begin(), workers.end(),
                        [] (const auto& worker) { return worker.active(); }));
        }
}

thread::pool_t::pool_t() :
        pool_t(thread::n_threads())
{
}

thread::pool_t::pool_t(std::size_t active_threads)
{
        const auto n_workers = static_cast<std::size_t>(thread::n_threads());
        active_threads = std::max(std::size_t(1), active_threads);

        for (size_t i = 0; i < n_workers; ++ i)
        {
                m_workers.emplace_back(m_tasks, i < active_threads);
        }
        for (size_t i = 0; i < n_workers; ++ i)
        {
                m_threads.emplace_back(std::ref(m_workers[i]));
        }

        assert(active_threads == thread::n_active_workers(m_workers));
}

thread::pool_t::~pool_t()
{
        // stop & join
        {
                const std::lock_guard<std::mutex> lock(m_tasks.m_mutex);

                m_tasks.m_stop = true;
                m_tasks.m_condition.notify_all();
        }

        for (auto& thread : m_threads)
        {
                thread.join();
        }
}

void thread::pool_t::wait()
{
        // wait for all tasks to be taken and the workers to finish
        std::unique_lock<std::mutex> lock(m_tasks.m_mutex);

        assert(thread::n_active_workers(m_workers) > 0);

        m_tasks.m_condition.wait(lock, [&] () 
        { 
                return m_tasks.m_tasks.empty() && m_tasks.m_running == 0;
        });
}

void thread::pool_t::activate(std::size_t count)
{
        const std::lock_guard<std::mutex> lock(m_tasks.m_mutex);

        count = std::max(std::size_t(1), std::min(count, n_workers()));

        std::size_t crt_count = thread::n_active_workers(m_workers);
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

        assert(count == thread::n_active_workers(m_workers));

        m_tasks.m_condition.notify_all();
}

std::size_t thread::pool_t::n_workers() const
{
        return m_workers.size();
}

std::size_t thread::pool_t::n_active_workers() const
{
        const std::lock_guard<std::mutex> lock(m_tasks.m_mutex);

        return thread::n_active_workers(m_workers);
}

std::size_t thread::pool_t::n_tasks() const
{
        const std::lock_guard<std::mutex> lock(m_tasks.m_mutex);

        return m_tasks.m_tasks.size();
}
