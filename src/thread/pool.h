#pragma once

#include "queue.h"
#include "worker.h"
#include <thread>

namespace thread
{
        ///
        /// \brief thread pool
        ///
        /// NB: this is heavily copied/inspired by http://progsch.net/wordpress/?p=81
        ///
        class NANO_PUBLIC pool_t
        {
        public:

                ///
                /// \brief single instance
                ///
                static pool_t& instance();

                ///
                /// \brief disable copying
                ///
                pool_t(const pool_t&) = delete;
                pool_t& operator=(const pool_t&) = delete;

                ///
                /// \brief disable moving
                ///
                pool_t(pool_t&&) = delete;
                pool_t& operator=(pool_t&&) = delete;

                ///
                /// \brief destructor
                ///
                ~pool_t();

                ///
                /// \brief set the given number of active workers [1, n_workers]
                ///
                void activate(const std::size_t count);

                ///
                /// \brief enqueue a new task to execute
                ///
                template<class F>
                auto enqueue(F f)
                {
                        return m_queue.enqueue(f);
                }

                ///
                /// \brief number of available worker threads
                ///
                std::size_t n_workers() const;

                ///
                /// \brief number of active worker threads
                ///
                std::size_t n_active_workers() const;

                ///
                /// \brief number of tasks still enqueued
                ///
                std::size_t n_tasks() const;

        private:

                ///
                /// \brief constructor (all available threads are active by default)
                ///
                pool_t();

        private:

                // attributes
                std::vector<std::thread>        m_threads;      ///<
                std::vector<worker_t>           m_workers;      ///<
                queue_t                         m_queue;        ///< tasks to execute + synchronization
        };
}
