#include "accumulator.h"
#include "common/thread_loop.hpp"
#include "loss.h"
#include <cassert>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        accumulator_t::accumulator_t(type t, source s, regularizer r, scalar_t lambda)
                :       m_type(t),
                        m_source(s),
                        m_regularizer(r),
                        m_lambda(lambda)
        {
                reset();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        accumulator_t::accumulator_t(const model_t& model, type t, source s, regularizer r, scalar_t lambda)
                :       m_type(t),
                        m_source(s),
                        m_regularizer(r),
                        m_lambda(lambda)
        {
                reset(model);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::reset(const model_t& model)
        {
                m_model = model.clone();
                m_vgrad.resize(dimensions());
                m_param = model.params();

                reset();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::reset()
        {
                m_value = 0.0;
                m_error = 0.0;
                m_count = 0;
                m_vgrad.setZero();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::reset(const vector_t& param)
        {
                assert(m_model);
                m_model->load_params(param);
                m_param = param;

                reset();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void accumulator_t::reset(type t, source s, regularizer r, scalar_t lambda)
        {
                m_type = t;
                m_source = s;
                m_regularizer = r;
                m_lambda = lambda;

                reset();
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

        void accumulator_t::cumulate(const vector_t& output, const vector_t& target, const loss_t& loss)
        {
                assert(m_model);
                assert(static_cast<size_t>(output.size()) == m_model->n_outputs());
                assert(static_cast<size_t>(target.size()) == m_model->n_outputs());

                switch (m_type)
                {
                case type::value:
                        break;

                case type::vgrad:
                        {
                                vector_t grad_params;
                                vector_t grad_inputs;
                                m_model->gradient(loss.vgrad(target, output), grad_params, grad_inputs);

                                switch (m_source)
                                {
                                case source::params:
                                        m_vgrad += grad_params;
                                        break;

                                case source::inputs:
                                        m_vgrad += grad_inputs;
                                        break;
                                }
                        }
                        break;
                }

                m_value += loss.value(target, output);
                m_error += loss.error(target, output);
                m_count ++;
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
                                assert(m_model);
                                data.reset(m_type, m_source, m_regularizer, m_lambda);
                                data.reset(*m_model);
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
                                assert(m_model);
                                data.reset(m_type, m_source, m_regularizer, m_lambda);
                                data.reset(*m_model);
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
                                assert(m_model);
                                data.reset(m_type, m_source, m_regularizer, m_lambda);
                                data.reset(*m_model);
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

        accumulator_t& accumulator_t::operator+=(const accumulator_t& other)
        {
                m_value += other.m_value;
                m_error += other.m_error;
                m_vgrad += other.m_vgrad;
                m_count += other.m_count;
                return *this;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t accumulator_t::value() const
        {
                return m_value / count();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t accumulator_t::error() const
        {
                return m_error / count();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        vector_t accumulator_t::vgrad() const
        {
                return m_vgrad / count();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t accumulator_t::dimensions() const
        {
                assert(m_model);

                switch (m_source)
                {
                case source::params:
                        return m_model->n_parameters();

                case source::inputs:
                default:
                        return m_model->n_inputs();
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
