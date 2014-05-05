#include "accumulator.h"
#include "common/thread_loop.hpp"
#include "common/math.hpp"
#include "loss.h"
#include <cassert>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        accumulator_t::accumulator_t(const model_t& model, type t, regularizer r, scalar_t lambda)
                :       accumulator_t(model.clone(), t, r, lambda)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        accumulator_t::accumulator_t(const rmodel_t& model, type t, regularizer r, scalar_t lambda)
                :       m_settings(t, r, lambda),
                        m_model(model),
                        m_data(dimensions())
        {
                m_data.m_params = model->params();
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
                m_model->load_params(params);
                m_data.m_params = params;

                reset();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::reset(type t, regularizer r, scalar_t lambda)
        {
                m_settings = settings_t(t, r, lambda);

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
                const vector_t output = m_model->value(image, sample.m_region);

                cumulate(output, target, loss);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update(const tensor_t& input, const vector_t& target, const loss_t& loss)
        {
                assert(m_model);
                const vector_t output = m_model->value(input);

                cumulate(output, target, loss);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update(const vector_t& input, const vector_t& target, const loss_t& loss)
        {
                assert(m_model);
                const vector_t output = m_model->value(input);

                cumulate(output, target, loss);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update(const task_t& task, const samples_t& samples, const loss_t& loss)
        {
                for (size_t i = 0; i < samples.size(); i ++)
                {
                        update(task, samples[i], loss);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update(const tensors_t& inputs, const vectors_t& targets, const loss_t& loss)
        {
                for (size_t i = 0; i < inputs.size(); i ++)
                {
                        update(inputs[i], targets[i], loss);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update(const vectors_t& inputs, const vectors_t& targets, const loss_t& loss)
        {
                for (size_t i = 0; i < inputs.size(); i ++)
                {
                        update(inputs[i], targets[i], loss);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update_mt(const task_t& task, const samples_t& samples, const loss_t& loss, size_t nthreads)
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

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update_mt(const tensors_t& inputs, const vectors_t& targets, const loss_t& loss, size_t nthreads)
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

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::update_mt(const vectors_t& inputs, const vectors_t& targets, const loss_t& loss, size_t nthreads)
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

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::cumulate(const vector_t& output, const vector_t& target, const loss_t& loss)
        {
                assert(m_model);
                assert(static_cast<size_t>(output.size()) == m_model->n_outputs());
                assert(static_cast<size_t>(target.size()) == m_model->n_outputs());

                const scalar_t value = loss.value(target, output);
                const scalar_t error = loss.error(target, output);

                // loss gradient
                switch (m_settings.m_type)
                {
                case type::value:
                        break;

                case type::vgrad:
                        {
                                vector_t grad_params, grad_inputs;
                                m_model->gradient(loss.vgrad(target, output), grad_params, grad_inputs);

                                switch (m_settings.m_regularizer)
                                {
                                case regularizer::none:
                                case regularizer::l2norm:
                                        m_data.m_grad1 += grad_params;
                                        break;

                                case regularizer::variational:
                                default:
                                        m_data.m_grad1 += grad_params;
                                        m_data.m_grad2 += value * grad_params;
                                }
                        }
                        break;
                }

                // loss value
                switch (m_settings.m_regularizer)
                {
                case regularizer::none:
                case regularizer::l2norm:
                        m_data.m_value1 += value;
                        m_data.m_error += error;
                        m_data.m_count ++;
                        break;

                case regularizer::variational:
                default:
                        m_data.m_value1 += value;
                        m_data.m_value2 += value * value;
                        m_data.m_error += error;
                        m_data.m_count ++;
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        accumulator_t& accumulator_t::operator+=(const accumulator_t& other)
        {
                m_data.m_value1 += other.m_data.m_value1;
                m_data.m_value2 += other.m_data.m_value2;
                m_data.m_grad1 += other.m_data.m_grad1;
                m_data.m_grad2 += other.m_data.m_grad2;
                m_data.m_error += other.m_data.m_error;
                m_data.m_count += other.m_data.m_count;
                return *this;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t accumulator_t::value() const
        {
                assert(count() > 0);
                assert(m_model);

                switch (m_settings.m_regularizer)
                {
                case regularizer::none:
                        return m_data.m_value1 / count();

                case regularizer::l2norm:
                        return m_data.m_value1 / count() +
                               0.5 * m_settings.m_lambda / dimensions() * m_data.m_params.squaredNorm();

                case regularizer::variational:
                default:// check my PhD thesis for the derivation!
                        return m_settings.m_lambda / count() * m_data.m_value2 +
                               (1.0 - m_settings.m_lambda) / math::square(count()) * math::square(m_data.m_value1);
                }
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

                switch (m_settings.m_regularizer)
                {
                case regularizer::none:
                        return m_data.m_grad1 / count();

                case regularizer::l2norm:
                        return m_data.m_grad1 / count() +
                               m_settings.m_lambda / dimensions() * m_data.m_params;

                case regularizer::variational:
                default:// check my PhD thesis for the derivation!
                        return 2.0 * m_settings.m_lambda / count() * m_data.m_grad2 +
                               2.0 * (1.0 - m_settings.m_lambda) / math::square(count()) * m_data.m_value1 * m_data.m_grad1;
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t accumulator_t::dimensions() const
        {
                assert(m_model);
                return m_model->n_parameters();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t accumulator_t::count() const
        {
                return m_data.m_count;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
