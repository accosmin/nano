#pragma once

#include "arch.h"
#include "queue.h"
#include "worker.h"
#include "section.h"
#include <thread>

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

                // attributes
                std::vector<std::thread>        m_threads;      ///<
                std::vector<worker_t>           m_workers;      ///<
                worker_queue_t                  m_queue;        ///< tasks to execute + synchronization
        };
}
