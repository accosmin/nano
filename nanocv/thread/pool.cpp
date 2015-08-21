#include "pool.h"
#include "thread.h"

namespace ncv
{
        thread::pool_t::pool_t(std::size_t nthreads)
        {
                nthreads = (nthreads == 0) ? ncv::n_threads() :
                                             std::max(size_t(1), std::min(nthreads, ncv::max_n_threads()));

                for (size_t i = 0; i < nthreads; i ++)
                {
                        m_workers.emplace_back(thread::pool_worker_t(m_tasks));
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

                for (size_t i = 0; i < m_workers.size(); i ++)
                {
                        m_workers[i].join();
                }
        }

        void thread::pool_t::wait()
        {
                // wait for all tasks to be taken and the workers to finish
                std::unique_lock<std::mutex> lock(m_tasks.m_mutex);

                while (!m_tasks.m_tasks.empty() || m_tasks.m_running > 0)
                {
                        m_tasks.m_condition.wait(lock);
                }
        }

        std::size_t thread::pool_t::n_workers() const
        {
                return m_workers.size();
        }

        std::size_t thread::pool_t::n_tasks() const
        {
                const std::lock_guard<std::mutex> lock(m_tasks.m_mutex);
                return m_tasks.m_tasks.size();
        }
}


