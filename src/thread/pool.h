#pragma once

#include "tasks.h"
#include "worker.h"
#include <vector>
#include <thread>

namespace thread
{
        ///
        /// \brief thread pool
        ///
        /// NB: this is heavily copied/inspired by http://progsch.net/wordpress/?p=81
        ///
        class NANOCV_PUBLIC pool_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit pool_t(std::size_t nthreads = 0);

                ///
                /// \brief disable copying
                ///
                pool_t(const pool_t&) = delete;
                pool_t& operator=(const pool_t&) = delete;

                ///
                /// \brief destructor
                ///
                ~pool_t();

                ///
                /// \brief movable
                ///
                pool_t(pool_t&&) = default;

                ///
                /// \brief movable
                ///
                pool_t& operator=(pool_t&&) = default;

                ///
                /// \brief enqueue a new task to execute
                ///
                template<class F>
                void enqueue(F f)
                {
                        m_tasks.enqueue(f);
                }

                ///
                /// \brief wait for all workers to finish running the tasks
                ///
                void wait();

                ///
                /// \brief number of available worker threads
                ///
                std::size_t n_workers() const;

                ///
                /// \brief number of tasks to run
                ///
                std::size_t n_tasks() const;

        private:

                // attributes
                std::vector<std::thread>        m_workers;      ///< worker threads
                tasks_t                         m_tasks;        ///< tasks to execute + synchronization
        };
}
