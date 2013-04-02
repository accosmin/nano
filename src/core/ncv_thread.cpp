#include "ncv_thread.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        void thread_impl::worker_pool::worker::operator()()
        {
                thread_impl::task_t task;
                while (true)
                {
                        // Wait for a new task to be available in the queue
                        {
                                thread_impl::lock_t lock(m_data.m_mutex);

                                while (!m_data.m_stop && m_data.m_tasks.empty())
                                {
                                        m_data.m_condition.wait(lock);
                                }

                                if (m_data.m_stop)
                                {
                                        break;
                                }

                                task = m_data.m_tasks.front();
                                m_data.m_tasks.pop_front();
                                m_data.m_running ++;
                        }

                        // Execute the task
                        task();

                        // Announce that a task was completed
                        {
                                thread_impl::lock_t lock(m_data.m_mutex);

                                m_data.m_running --;
                                m_data.m_condition.notify_all();
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        thread_impl::worker_pool::worker_pool()
                :       m_data()
        {
                for (size_t i = 0; i < thread_impl::n_threads(); i ++)
                {
                        m_workers.push_back(std::thread(thread_impl::worker_pool::worker(m_data)));
                }
        }

        //-------------------------------------------------------------------------------------------------

        thread_impl::worker_pool::~worker_pool()
        {
                // Stop & join
                m_data.m_stop = true;
                m_data.m_condition.notify_all();

                for (size_t i = 0; i < m_workers.size(); i ++)
                {
                        m_workers[i].join();
                }
        }

        //-------------------------------------------------------------------------------------------------

        void thread_impl::worker_pool::wait()
        {
                // Wait for all tasks to be taken and the workers to finish
                thread_impl::lock_t lock(m_data.m_mutex);

                while (!m_data.m_tasks.empty() || m_data.m_running > 0)
                {
                        m_data.m_condition.wait(lock);
                }
        }

        //-------------------------------------------------------------------------------------------------
}


