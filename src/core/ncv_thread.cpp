#include "ncv_thread.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        namespace thread_impl
        {
                typename worker_pool_t::this_instance_t    worker_pool_t::m_instance = nullptr;
                typename worker_pool_t::this_mutex_t       worker_pool_t::m_once_flag;
        }

        //-------------------------------------------------------------------------------------------------

        void thread_impl::worker_pool_t::worker::operator()()
        {
                thread_impl::task_t task;
                while (true)
                {
                        // wait for a new task to be available in the queue
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

                        // execute the task
                        task();

                        // announce that a task was completed
                        {
                                thread_impl::lock_t lock(m_data.m_mutex);

                                m_data.m_running --;
                                m_data.m_condition.notify_all();
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        thread_impl::worker_pool_t::worker_pool_t()
                :       m_data()
        {
                for (size_t i = 0; i < thread_impl::n_threads(); i ++)
                {
                        m_workers.push_back(std::thread(thread_impl::worker_pool_t::worker(m_data)));
                }
        }

        //-------------------------------------------------------------------------------------------------

        thread_impl::worker_pool_t::~worker_pool_t()
        {
                // stop & join
                m_data.m_stop = true;
                m_data.m_condition.notify_all();

                for (size_t i = 0; i < m_workers.size(); i ++)
                {
                        m_workers[i].join();
                }
        }

        //-------------------------------------------------------------------------------------------------

        void thread_impl::worker_pool_t::wait()
        {
                // wait for all tasks to be taken and the workers to finish
                thread_impl::lock_t lock(m_data.m_mutex);

                while (!m_data.m_tasks.empty() || m_data.m_running > 0)
                {
                        m_data.m_condition.wait(lock);
                }
        }

        //-------------------------------------------------------------------------------------------------
}


