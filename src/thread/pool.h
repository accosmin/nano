#pragma once

#include "arch.h"
#include "queue.h"
#include "worker.h"
#include "section.h"
#include <thread>
#include <cassert>
#include <algorithm>

namespace nano
{
        ///
        /// \brief thread pool.
        /// NB: this is heavily copied/inspired by http://progsch.net/wordpress/?p=81
        ///
        class NANO_PUBLIC thread_pool_t
        {
        public:

                ///
                /// \brief single instance
                ///
                static thread_pool_t& instance();

                ///
                /// \brief disable copying
                ///
                thread_pool_t(const thread_pool_t&) = delete;
                thread_pool_t& operator=(const thread_pool_t&) = delete;

                ///
                /// \brief disable moving
                ///
                thread_pool_t(thread_pool_t&&) = delete;
                thread_pool_t& operator=(thread_pool_t&&) = delete;

                ///
                /// \brief destructor
                ///
                ~thread_pool_t();

                ///
                /// \brief set the given number of active workers [1, n_workers]
                ///
                void activate(const std::size_t count);

                ///
                /// \brief enqueue a new task to execute
                ///
                template <typename tfunction>
                auto enqueue(tfunction f)
                {
                        return m_queue.enqueue(f);
                }

                ///
                /// \brief number of available worker threads
                ///
                std::size_t workers() const;

                ///
                /// \brief number of active worker threads
                ///
                std::size_t active_workers() const;

                ///
                /// \brief number of tasks still enqueued
                ///
                std::size_t tasks() const;

        private:

                ///
                /// \brief constructor (all available threads are active by default)
                ///
                thread_pool_t();

                ///
                /// \brief returns the number of active workers
                ///
                static std::size_t active_workers(const std::vector<worker_t>& workers)
                {
                        return  static_cast<std::size_t>(std::count_if(workers.begin(), workers.end(),
                                [] (const auto& worker) { return worker.active(); }));
                }

        private:

                // attributes
                std::vector<std::thread>        m_threads;      ///<
                std::vector<worker_t>           m_workers;      ///<
                worker_queue_t                  m_queue;        ///< tasks to execute + synchronization
        };

        #include <cassert>
        #include <algorithm>

        thread_pool_t& thread_pool_t::instance()
        {
                static thread_pool_t the_pool;
                return the_pool;
        }

        thread_pool_t::thread_pool_t()
        {
                const auto n_workers = static_cast<std::size_t>(logical_cpus());
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

                assert(n_active_threads == active_workers(m_workers));
        }

        thread_pool_t::~thread_pool_t()
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

        void thread_pool_t::activate(std::size_t count)
        {
                const std::lock_guard<std::mutex> lock(m_queue.m_mutex);

                count = std::max(std::size_t(1), std::min(count, workers()));

                std::size_t crt_count = active_workers(m_workers);
                assert(crt_count > 0);
                for (auto& worker : m_workers)
                {
                        if (crt_count == count)
                        {
                                break;
                        }

                        else if (crt_count > count && worker.deactivate())
                        {
                                -- crt_count;
                        }

                        else if (crt_count < count && worker.activate())
                        {
                                ++ crt_count;
                        }
                }

                assert(count == active_workers(m_workers));

                m_queue.m_condition.notify_all();
        }

        std::size_t thread_pool_t::workers() const
        {
                return m_workers.size();
        }

        std::size_t thread_pool_t::active_workers() const
        {
                const std::lock_guard<std::mutex> lock(m_queue.m_mutex);

                return active_workers(m_workers);
        }

        std::size_t thread_pool_t::tasks() const
        {
                const std::lock_guard<std::mutex> lock(m_queue.m_mutex);

                return m_queue.m_tasks.size();
        }
}
