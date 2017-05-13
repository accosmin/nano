#include "function.h"
#include "accumulator.h"

namespace nano
{
        struct batch_function_t final : public function_t
        {
                batch_function_t(accumulator_t& acc, const iterator_t& iterator, const task_t& task, const fold_t& fold) :
                        function_t("ml optimization function", acc.psize(), acc.psize(), acc.psize(), convexity::no, 1e+6),
                        m_accumulator(acc),
                        m_iterator(iterator),
                        m_task(task),
                        m_fold(fold)
                {
                }

                scalar_t vgrad(const vector_t& x, vector_t* gx) const override
                {
                        m_accumulator.params(x);
                        m_accumulator.mode(gx ? accumulator_t::type::vgrad : accumulator_t::type::value);
                        m_accumulator.update(m_iterator, m_task, m_fold);
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
        };
}
