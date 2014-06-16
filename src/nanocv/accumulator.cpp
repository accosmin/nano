#include "accumulator.h"
#include "loss.h"
#include <cassert>

namespace ncv
{        
        accumulator_t::accumulator_t(const model_t& model, size_t nthreads, type t, scalar_t lambda)
                :       m_pool(nthreads),
                        m_models(m_pool.n_workers()),
                        m_configs(m_pool.n_workers(), { t, lambda }),
                        m_datas(m_pool.n_workers(), { model.psize() }),
                        
                        m_config(t, lambda),
                        m_data(model.psize())
        {
                m_data.m_params = model.params();
                
                for (size_t th = 0; th < m_pool.n_workers(); th ++)
                {                
                        m_models[th] = model.clone();
                        m_datas[th].m_params = m_data.m_params;
                }                
        }

        void accumulator_t::reset(const vector_t& params)
        {
                m_data.m_params = params;
                
                for (size_t th = 0; th < m_pool.n_workers(); th ++)
                {                
                        m_models[th]->load_params(params);
                }                

                reset();
        }

        void accumulator_t::reset()
        {
                m_data.reset();
                
                for (size_t th = 0; th < m_pool.n_workers(); th ++)
                {
                        m_datas[th].reset();
                }
        }
        
        void accumulator_t::cumulate(
                const model_t& model, const vector_t& output, const vector_t& target, const loss_t& loss,
                const settings_t& config, data_t& data)
        {
                assert(static_cast<size_t>(output.size()) == model.osize());
                assert(static_cast<size_t>(target.size()) == model.osize());
                
                // loss gradient
                switch (config.m_type)
                {
                case type::value:
                        break;
                        
                case type::vgrad:
                        data.m_vgrad += model.gradient(loss.vgrad(target, output));
                        break;
                }
                
                // loss value
                data.m_value += loss.value(target, output);
                data.m_error += loss.error(target, output);
                data.m_count ++;
        }

        void accumulator_t::update(const task_t& task, const sample_t& sample, const loss_t& loss)
        {
                assert(sample.m_index < task.n_images());

                const image_t& image = task.image(sample.m_index);
                const vector_t& target = sample.m_target;
                
                const model_t& model = *m_models.begin()->get();                                
                const vector_t& output = model.forward(image, sample.m_region).vector();

                cumulate(model, output, target, loss, m_config, m_data);
        }

        void accumulator_t::update(const tensor_t& input, const vector_t& target, const loss_t& loss)
        {
                const model_t& model = *m_models.begin()->get();
                const vector_t& output = model.forward(input).vector();

                cumulate(model, output, target, loss, m_config, m_data);
        }

        void accumulator_t::update(const vector_t& input, const vector_t& target, const loss_t& loss)
        {
                const model_t& model = *m_models.begin()->get();
                const vector_t& output = model.forward(input).vector();

                cumulate(model, output, target, loss, m_config, m_data);
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
                        thread_loop_cumulate<size_t>
                        (
                                samples.size(),
                                [&] (accumulator_t& data)
                                {
                                        data = *this;
                                },
                                [&] (size_t i, accumulator_t& data)
                                {
                                        data.update(task, samples[i], loss);
                                },
                                [&] (accumulator_t& data)
                                {
                                        this->operator +=(data);
                                },
                                m_pool
                        );
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
                        thread_loop_cumulate<accumulator_t>
                        (
                                inputs.size(),
                                [&] (accumulator_t& data)
                                {
                                        data = *this;
                                },
                                [&] (size_t i, accumulator_t& data)
                                {
                                        data.update(inputs[i], targets[i], loss);
                                },
                                [&] (accumulator_t& data)
                                {
                                        this->operator +=(data);
                                },
                                nthreads
                        );
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
                        thread_loop_cumulate<accumulator_t>
                        (
                                inputs.size(),
                                [&] (accumulator_t& data)
                                {
                                        data = *this;
                                },
                                [&] (size_t i, accumulator_t& data)
                                {
                                        data.update(inputs[i], targets[i], loss);
                                },
                                [&] (accumulator_t& data)
                                {
                                        this->operator +=(data);
                                },
                                nthreads
                        );
                }
        }
        
        scalar_t accumulator_t::value() const
        {
                assert(count() > 0);

                return  m_data.m_value / count() +
                        0.5 * m_config.m_lambda / dimensions() * m_data.m_params.squaredNorm();
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
                        m_config.m_lambda / dimensions() * m_data.m_params;
        }

        size_t accumulator_t::dimensions() const
        {
                return m_data.m_params.size();
        }

        size_t accumulator_t::count() const
        {
                return m_data.m_count;
        }

        scalar_t accumulator_t::lambda() const
        {
                return m_config.m_lambda;
        }
}
	
