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
                /// \brief number of tasks still enqueued
                ///
                std::size_t tasks() const;

        private:

                ///
                /// \brief constructor (all available threads are active by default)
                ///
                thread_pool_t();

        private:

                // attributes
                std::vector<std::thread>        m_threads;      ///<
                std::vector<worker_t>           m_workers;      ///<
                worker_queue_t                  m_queue;        ///< tasks to execute + synchronization
        };

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

        std::size_t thread_pool_t::workers() const
        {
                return m_workers.size();
        }

        std::size_t thread_pool_t::tasks() const
        {
                const std::lock_guard<std::mutex> lock(m_queue.m_mutex);

                return m_queue.m_tasks.size();
        }
}
