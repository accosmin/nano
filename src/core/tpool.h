#pragma once

#include "arch.h"
#include <deque>
#include <mutex>
#include <future>
#include <thread>
#include <vector>
#include <condition_variable>

namespace nano
{
        using future_t = std::future<void>;
        using worker_task_t = std::packaged_task<void()>;

        ///
        /// \brief queue tasks to be run in a thread pool.
        ///
        class worker_queue_t
        {
        public:
                ///
                /// \brief constructor
                ///
                worker_queue_t() = default;

                ///
                /// \brief enqueue a new task to execute
                ///
                template <typename tfunction>
                future_t enqueue(tfunction f)
                {
                        auto task = worker_task_t(f);
                        auto fut = task.get_future();

                        const std::lock_guard<std::mutex> lock(m_mutex);
                        m_tasks.emplace_back(std::move(task));
                        m_condition.notify_all();

                        return fut;
                }

                // attributes
                std::deque<worker_task_t>       m_tasks;                ///< tasks to execute
                mutable std::mutex              m_mutex;                ///< synchronization
                mutable std::condition_variable m_condition;            ///< signaling
                bool                            m_stop{false};          ///< stop requested
        };

        ///
        /// \brief worker to process tasks enqueued in a thread pool.
        ///
        class worker_t
        {
        public:
                ///
                /// \brief constructor
                ///
                explicit worker_t(worker_queue_t& queue) : m_queue(queue) {}

                ///
                /// \brief execute tasks when available
                ///
                void operator()() const
                {
                        while (true)
                        {
                                worker_task_t task;

                                // wait for a new task to be available in the queue
                                {
                                        std::unique_lock<std::mutex> lock(m_queue.m_mutex);

                                        m_queue.m_condition.wait(lock, [&]
                                        {
                                                return m_queue.m_stop || !m_queue.m_tasks.empty();
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

        private:

                // attributes
                worker_queue_t& m_queue;        ///< task queue to process
        };

        ///
        /// \brief RAII object to wait for a given set of futures (aka barrier).
        ///
        template <typename tfuture>
        class section_t
        {
        public:
                ///
                /// \brief destructor
                ///
                ~section_t()
                {
                        // block until all futures are done
                        for (const auto& future : m_futures)
                        {
                                future.wait();
                        }
                }

                ///
                /// \brief add a new future to wait for.
                ///
                void push_back(tfuture future)
                {
                        m_futures.emplace_back(std::move(future));
                }

        private:

                // attributes
                std::vector<tfuture>    m_futures;
        };

        ///
        /// \brief thread pool.
        /// NB: this is heavily copied/inspired by http://progsch.net/wordpress/?p=81
        ///
        class thread_pool_t
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
                /// \brief constructor
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

                m_workers.reserve(n_workers);
                for (size_t i = 0; i < n_workers; ++ i)
                {
                        m_workers.emplace_back(m_queue);
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
