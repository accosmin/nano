#include "function.h"
#include "minibatch.h"
#include "accumulator.h"

namespace nano
{
        struct stoch_function_t final : public function_t
        {
                stoch_function_t(accumulator_t& acc, const iterator_t& iterator, const task_t& task, minibatch_t& minibatch) :
                        function_t("ml optimization function", acc.psize(), acc.psize(), acc.psize(), convexity::no, 1e+6),
                        m_accumulator(acc),
                        m_iterator(iterator),
                        m_task(task),
                        m_fold(minibatch.fold()),
                        m_minibatch(minibatch)
                {
                }

                size_t stoch_ratio() const override
                {
                        const auto batch_size = m_task.size(m_fold);
                        const auto stoch_size = m_minibatch.size();
                        assert(stoch_size > 0);
                        return nano::idiv(batch_size, stoch_size);
                }

                void stoch_next() const override
                {
                        // next minibatch
                        m_minibatch.next();
                }

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override
                {
                        m_accumulator.params(x);
                        m_accumulator.mode(gx ? accumulator_t::type::vgrad : accumulator_t::type::value);
                        m_accumulator.update(m_iterator, m_task, m_fold);
                        return get(gx);
                }

                scalar_t stoch_vgrad(const vector_t& x, vector_t* gx) const override
                {
                        m_accumulator.params(x);
                        m_accumulator.mode(gx ? accumulator_t::type::vgrad : accumulator_t::type::value);
                        m_accumulator.update(m_iterator, m_task, m_fold, m_minibatch.begin(), m_minibatch.end());
                        return get(gx);
                }

                scalar_t get(vector_t* gx) const
                {
                        if (gx)
                        {
                                *gx = m_accumulator.vgrad();
                        }
                        return m_accumulator.vstats().avg();
                }

                // attributes
                accumulator_t&          m_accumulator;  ///< function value and gradient accumulator
                const iterator_t&       m_iterator;     ///<
                const task_t&           m_task;         ///<
                const fold_t            m_fold;         ///<
                minibatch_t&            m_minibatch;    ///<
        };
}
