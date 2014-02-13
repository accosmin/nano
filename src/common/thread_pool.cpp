#include "thread_pool.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        void thread_pool_t::worker_t::operator()()
        {
                task_t task;
                while (true)
                {
                        // wait for a new task to be available in the queue
                        {
                                lock_t lock(m_data.m_mutex);

                                while ( !m_data.m_stop &&
                                        m_data.m_tasks.empty())
                                {
                                        m_data.m_condition.wait(lock);
                                }

                                if (m_data.m_stop)
                                {
                                        m_data.m_running = 0;
                                        m_data.m_tasks.clear();
                                        m_data.m_condition.notify_all();
                                        break;
                                }

                                else if (m_data.m_running >= m_data.m_maxrunning)
                                {
                                        continue;
                                }

                                task = m_data.m_tasks.front();
                                m_data.m_tasks.pop_front();
                                m_data.m_running ++;
                        }

                        // execute the task
                        task();

                        // announce that a task was completed
                        {
                                lock_t lock(m_data.m_mutex);

                                m_data.m_running --;
                                m_data.m_condition.notify_all();
                        }
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        thread_pool_t::thread_pool_t()
                :       m_data(ncv::n_threads())
        {
                for (size_t i = 0; i < ncv::n_threads(); i ++)
                {
                        m_workers.push_back(std::thread(thread_pool_t::worker_t(m_data)));
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        thread_pool_t::~thread_pool_t()
        {
                // stop & join
                {
                        const lock_t lock(m_data.m_mutex);
                        m_data.m_stop = true;
                }
                m_data.m_condition.notify_all();

                for (size_t i = 0; i < m_workers.size(); i ++)
                {
                        m_workers[i].join();
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void thread_pool_t::wait()
        {
                // wait for all tasks to be taken and the workers to finish
                lock_t lock(m_data.m_mutex);

                while (!m_data.m_tasks.empty() || m_data.m_running > 0)
                {
                        m_data.m_condition.wait(lock);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void thread_pool_t::resize(size_t threads)
        {
                {
                        const lock_t lock(m_data.m_mutex);
                        m_data.m_maxrunning = (threads == 0) ? ncv::n_threads() : std::min(threads, ncv::n_threads());
                }
                m_data.m_condition.notify_one();
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}


