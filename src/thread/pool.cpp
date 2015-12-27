#include "pool.h"
#include "thread.h"
#include <algorithm>

namespace thread
{
        static std::size_t n_active_workers(const std::vector<worker_config_t>& settings)
        {
                return  static_cast<std::size_t>(std::count_if(settings.begin(), settings.end(),
                        [] (const auto& config) { return config.active(); }));
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
                m_configs.emplace_back(i < active_threads);
        }
        for (size_t i = 0; i < n_workers; ++ i)
        {
                m_workers.emplace_back(thread::worker_t(m_tasks, m_configs[i]));
        }
}

thread::pool_t::~pool_t()
{
        // stop & join
        {
                const std::lock_guard<std::mutex> lock(m_tasks.m_mutex);

                m_tasks.m_stop = true;
        }
        m_tasks.m_condition.notify_all();

        for (size_t i = 0; i < m_workers.size(); ++ i)
        {
                if (m_workers[i].joinable())
                {
                        m_workers[i].join();
                }
        }
}

void thread::pool_t::wait()
{
        // wait for all tasks to be taken and the workers to finish
        std::unique_lock<std::mutex> lock(m_tasks.m_mutex);

        m_tasks.m_condition.wait(lock, [&] () { return m_tasks.m_tasks.empty() && m_tasks.m_running == 0; });
}

void thread::pool_t::activate(std::size_t count)
{
        {
                const std::lock_guard<std::mutex> lock(m_tasks.m_mutex);

                count = std::max(std::size_t(1), std::min(count, n_workers()));

                std::size_t crt_count = thread::n_active_workers(m_configs);
                for (auto& config :  m_configs)
                {
                        if (crt_count == count)
                        {
                                break;
                        }

                        else if (crt_count > count)
                        {
                                if (config.deactivate())
                                {
                                        -- crt_count;
                                }
                        }

                        else if (crt_count < count)
                        {
                                if (config.activate())
                                {
                                        ++ crt_count;
                                }
                        }
                }
        }

        m_tasks.m_condition.notify_all();
}

std::size_t thread::pool_t::n_workers() const
{
        return m_workers.size();
}

std::size_t thread::pool_t::n_active_workers() const
{
        const std::lock_guard<std::mutex> lock(m_tasks.m_mutex);

        return thread::n_active_workers(m_configs);
}

std::size_t thread::pool_t::n_tasks() const
{
        const std::lock_guard<std::mutex> lock(m_tasks.m_mutex);

        return m_tasks.m_tasks.size();
}
