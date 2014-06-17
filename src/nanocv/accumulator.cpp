#include "accumulator.h"
#include "loss.h"
#include "common/thread_loop.hpp"
#include <cassert>

namespace ncv
{        
        void accumulator_t::data_t::cumulate(
                const vector_t& output, const vector_t& target, const loss_t& loss)
        {
                assert(static_cast<size_t>(output.size()) == m_model->osize());
                assert(static_cast<size_t>(target.size()) == m_model->osize());
                
                // loss value
                m_value += loss.value(target, output);
                m_error += loss.error(target, output);
                m_count ++;
                
                // loss gradient
                switch (m_config.m_type)
                {
                        case type::value:
                                break;
                                
                        case type::vgrad:
                                m_vgrad += m_model->gradient(loss.vgrad(target, output));
                                break;
                }
        }
        
        void accumulator_t::data_t::update(const task_t& task, const sample_t& sample, const loss_t& loss)
        {
                assert(sample.m_index < task.n_images());
                
                const image_t& image = task.image(sample.m_index);
                const vector_t& target = sample.m_target;                
                const vector_t& output = m_model->forward(image, sample.m_region).vector();
                
                cumulate(output, target, loss);
        }
        
        void accumulator_t::data_t::update(const tensor_t& input, const vector_t& target, const loss_t& loss)
        {
                const vector_t& output = m_model->forward(input).vector();
                
                cumulate(output, target, loss);
        }
        
        void accumulator_t::data_t::update(const vector_t& input, const vector_t& target, const loss_t& loss)
        {
                const vector_t& output = m_model->forward(input).vector();
                
                cumulate(output, target, loss);
        }
        
        accumulator_t::accumulator_t(const model_t& model, size_t nthreads, type t, scalar_t lambda)
                :       m_pool(nthreads),                        
                        m_datas(m_pool.n_workers(), { model.psize(), t, lambda }),
                        m_data(model.psize(), t, lambda)
        {
                m_data.reset(model);                
                for (data_t& data : m_datas)
                {
                        data.reset(model);
                }
        }

        void accumulator_t::reset(const vector_t& params)
        {
                m_data.reset(params);
                for (data_t& data : m_datas)
                {
                        data.reset(params);
                }
        }

        void accumulator_t::reset()
        {
                m_data.reset();
                for (data_t& data : m_datas)
                {
                        data.reset();
                }
        }       

        void accumulator_t::update(const task_t& task, const sample_t& sample, const loss_t& loss)
        {
                m_data.update(task, sample, loss);
        }

        void accumulator_t::update(const tensor_t& input, const vector_t& target, const loss_t& loss)
        {
                m_data.update(input, target, loss);
        }

        void accumulator_t::update(const vector_t& input, const vector_t& target, const loss_t& loss)
        {
                m_data.update(input, target, loss);
        }

        void accumulator_t::update(const task_t& task, const samples_t& samples, const loss_t& loss)
        {
                if (m_pool.n_workers() == 1)
                {
                        for (size_t i = 0; i < samples.size(); i ++)
                        {
                                update(task, samples[i], loss);
                        }
                }

                else
                {
                        thread_loopit
                        (
                                samples.size(),
                                [&] (size_t i, size_t th)
                                {
                                        m_datas[th].update(task, samples[i], loss);
                                },
                                m_pool
                        );
                        
                        for (const data_t& data : m_datas)
                        {
                                m_data += data;
                        }
                }
        }

        void accumulator_t::update(const tensors_t& inputs, const vectors_t& targets, const loss_t& loss)
        {
                if (m_pool.n_workers() == 1)
                {
                        for (size_t i = 0; i < inputs.size(); i ++)
                        {
                                update(inputs[i], targets[i], loss);
                        }
                }

                else
                {
                        thread_loopit
                        (
                                inputs.size(),
                                [&] (size_t i, size_t th)
                                {
                                        m_datas[th].update(inputs[i], targets[i], loss);
                                },
                                m_pool
                        );
                        
                        for (const data_t& data : m_datas)
                        {
                                m_data += data;
                        }
                }
        }

        void accumulator_t::update(const vectors_t& inputs, const vectors_t& targets, const loss_t& loss)
        {
                if (m_pool.n_workers() == 1)
                {
                        for (size_t i = 0; i < inputs.size(); i ++)
                        {
                                update(inputs[i], targets[i], loss);
                        }
                }

                else
                {
                        thread_loopit
                        (
                                inputs.size(),
                                [&] (size_t i, size_t th)
                                {
                                        m_datas[th].update(inputs[i], targets[i], loss);
                                },
                                m_pool
                        );
                        
                        for (const data_t& data : m_datas)
                        {
                                m_data += data;
                        }
                }
        }
        
        scalar_t accumulator_t::value() const
        {
                assert(count() > 0);

                return  m_data.m_value / count() +
                        0.5 * m_data.m_config.m_lambda / dimensions() * m_data.m_params.squaredNorm();
        }

        scalar_t accumulator_t::error() const
        {
                assert(count() > 0);

                return m_data.m_error / count();
        }

        vector_t accumulator_t::vgrad() const
        {
                assert(count() > 0);

                return  m_data.m_vgrad / count() +
                        m_data.m_config.m_lambda / dimensions() * m_data.m_params;
        }

        size_t accumulator_t::dimensions() const
        {
                return static_cast<size_t>(m_data.m_params.size());
        }

        size_t accumulator_t::count() const
        {
                return m_data.m_count;
        }

        scalar_t accumulator_t::lambda() const
        {
                return m_data.m_config.m_lambda;
        }
}
	
