#include "task.h"
#include "accumulator.h"
#include "trainer_data.h"
#include "cortex/util/logger.h"

namespace zob
{
        trainer_data_t::trainer_data_t(const task_t& task, const fold_t& fold, const loss_t& loss, const vector_t& x0,
                        accumulator_t& lacc,
                        accumulator_t& gacc)
                :       m_task(task),
                        m_tsampler(task.samples()),
                        m_vsampler(task.samples()),
                        m_loss(loss),
                        m_x0(x0),
                        m_lacc(lacc),
                        m_gacc(gacc)
        {
                // create training & validation samples
                m_tsampler.push(fold);
                m_tsampler.push(zob::annotation::annotated);

                m_tsampler.split(80, m_vsampler);

                if (m_tsampler.empty() || m_vsampler.empty())
                {
                        const string_t message = "no annotated training samples!";
                        log_error() << message;
                        throw std::runtime_error(message);
                }
        }

        void trainer_data_t::set_lambda(scalar_t lambda) const
        {
                m_lacc.set_lambda(lambda);
                m_gacc.set_lambda(lambda);
        }

        scalar_t trainer_data_t::lambda() const
        {
                return m_lacc.lambda();
        }

        size_t trainer_data_t::epoch_size(const size_t batchsize) const
        {
                return (m_tsampler.size() + batchsize - 1) / batchsize;
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
                        data.m_lacc.set_params(x);
                        data.m_lacc.update(data.m_task, data.m_tsampler.get(), data.m_loss);

                        return data.m_lacc.value();
                };
        }

        opt_opgrad_t make_opgrad(const trainer_data_t& data)
        {
                return [&] (const vector_t& x, vector_t& gx)
                {
                        data.m_gacc.set_params(x);
                        data.m_gacc.update(data.m_task, data.m_tsampler.get(), data.m_loss);

                        gx = data.m_gacc.vgrad();
                        return data.m_gacc.value();
                };
        }
}

