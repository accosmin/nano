#include "trainer_data.h"
#include "accumulator.h"
#include "sampler.h"

namespace ncv
{
        trainer_data_t::trainer_data_t(const task_t& task,
                       const sampler_t& tsampler,
                       const sampler_t& vsampler,
                       const loss_t& loss,
                       const vector_t& x0,
                       accumulator_t& lacc,
                       accumulator_t& gacc)
                :       m_task(task),
                        m_tsampler(tsampler),
                        m_vsampler(vsampler),
                        m_loss(loss),
                        m_x0(x0),
                        m_lacc(lacc),
                        m_gacc(gacc)
        {
        }

        opt_opsize_t make_opsize(const trainer_data_t& data)
        {
                return [&] ()
                {
                        return data.m_gacc.psize();
                };
        }

        opt_opfval_t make_opfval(const trainer_data_t& data)
        {
                return [&] (const vector_t& x)
                {
                        data.m_lacc.reset(x);
                        data.m_lacc.update(data.m_task, data.m_tsampler.get(), data.m_loss);

                        return data.m_lacc.value();
                };
        }

        opt_opgrad_t make_opgrad(const trainer_data_t& data)
        {
                return [&] (const vector_t& x, vector_t& gx)
                {
                        data.m_gacc.reset(x);
                        data.m_gacc.update(data.m_task, data.m_tsampler.get(), data.m_loss);

                        gx = data.m_gacc.vgrad();
                        return data.m_gacc.value();
                };
        }
}
	
