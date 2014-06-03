#include "accumulator.h"
#include "common/thread_loop.hpp"
#include "common/math.hpp"
#include "loss.h"
#include <cassert>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        accumulator_t::accumulator_t(const model_t& model, type t, scalar_t lambda)
                :       accumulator_t(model.clone(), t, lambda)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        accumulator_t::accumulator_t(const rmodel_t& model, type t, scalar_t lambda)
                :       m_settings(t, lambda),
                        m_model(model),
                        m_data(model ? dimensions() : 0)
        {
                if (model)
                {
                        m_data.m_params = model->params();
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        accumulator_t::accumulator_t(const accumulator_t& other)
                :       m_settings(other.m_settings),
                        m_model(other.m_model->clone()),
                        m_data(other.m_data)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        accumulator_t& accumulator_t::operator=(const accumulator_t& other)
        {
                if (this != &other)
                {
                        assert(other.m_model);
                        m_settings = other.m_settings;
                        m_model = other.m_model->clone();
                        m_data = other.m_data;
                }

                return *this;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::reset(const model_t& model)
        {
                assert(m_model);
                m_model = model.clone();
                m_data.m_params = model.params();

                reset();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::reset(const vector_t& params)
        {
                assert(m_model);
                if (!m_model->load_params(params))
                {
                        std::cout << "accumulator_t::reset: params = " << params.size() << "/" << m_model->psize() << std::endl;
                }
                m_data.m_params = params;

                reset();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::reset(type t, scalar_t lambda)
        {
                m_settings = settings_t(t, lambda);

                reset();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::reset()
        {
                m_data.reset();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update(const task_t& task, const sample_t& sample, const loss_t& loss)
        {
                assert(m_model);
                assert(sample.m_index < task.n_images());

                const image_t& image = task.image(sample.m_index);
                const vector_t& target = sample.m_target;
                const vector_t& output = m_model->forward(image, sample.m_region).vector();

                cumulate(output, target, loss);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update(const tensor_t& input, const vector_t& target, const loss_t& loss)
        {
                assert(m_model);
                const vector_t& output = m_model->forward(input).vector();

                cumulate(output, target, loss);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update(const vector_t& input, const vector_t& target, const loss_t& loss)
        {
                assert(m_model);
                const vector_t& output = m_model->forward(input).vector();

                cumulate(output, target, loss);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update(const task_t& task, const samples_t& samples, const loss_t& loss, size_t nthreads)
        {
                if (nthreads == 1)
                {
                        for (size_t i = 0; i < samples.size(); i ++)
                        {
                                update(task, samples[i], loss);
                        }
                }

                else
                {
                        thread_loop_cumulate<accumulator_t>
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
                                nthreads
                        );
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update(const tensors_t& inputs, const vectors_t& targets, const loss_t& loss, size_t nthreads)
        {
                if (nthreads == 1)
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

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update(const vectors_t& inputs, const vectors_t& targets, const loss_t& loss, size_t nthreads)
        {
                if (nthreads == 1)
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

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::cumulate(const vector_t& output, const vector_t& target, const loss_t& loss)
        {
                assert(m_model);
                assert(static_cast<size_t>(output.size()) == m_model->osize());
                assert(static_cast<size_t>(target.size()) == m_model->osize());

                const scalar_t value = loss.value(target, output);
                const scalar_t error = loss.error(target, output);

                // loss gradient
                switch (m_settings.m_type)
                {
                case type::value:
                        break;

                case type::vgrad:
                        {
                                const vector_t grad = m_model->gradient(loss.vgrad(target, output));

                                m_data.m_vgrad += grad;
                        }
                        break;
                }

                // loss value
                m_data.m_value += value;
                m_data.m_error += error;
                m_data.m_count ++;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        accumulator_t& accumulator_t::operator+=(const accumulator_t& other)
        {
                m_data.m_value += other.m_data.m_value;
                m_data.m_vgrad += other.m_data.m_vgrad;
                m_data.m_error += other.m_data.m_error;
                m_data.m_count += other.m_data.m_count;
                return *this;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t accumulator_t::value() const
        {
                assert(count() > 0);
                assert(m_model);

                return  m_data.m_value / count() +
                        0.5 * m_settings.m_lambda / dimensions() * m_data.m_params.squaredNorm();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t accumulator_t::error() const
        {
                assert(count() > 0);

                return m_data.m_error / count();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        vector_t accumulator_t::vgrad() const
        {
                assert(count() > 0);

                return  m_data.m_vgrad / count() +
                        m_settings.m_lambda / dimensions() * m_data.m_params;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t accumulator_t::dimensions() const
        {
                assert(m_model);
                return m_model->psize();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t accumulator_t::count() const
        {
                return m_data.m_count;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t accumulator_t::lambda() const
        {
                return m_settings.m_lambda;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
