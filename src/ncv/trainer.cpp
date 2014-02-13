#include "trainer.h"
#include "common/thread_loop.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        trainer_state_t::trainer_state_t(size_t n_parameters)
                :       m_params(n_parameters),
                        m_tvalue(std::numeric_limits<scalar_t>::max()),
                        m_terror(std::numeric_limits<scalar_t>::max()),
                        m_vvalue(std::numeric_limits<scalar_t>::max()),
                        m_verror(std::numeric_limits<scalar_t>::max())
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool trainer_state_t::update(const vector_t& params,
                    scalar_t tvalue, scalar_t terror,
                    scalar_t vvalue, scalar_t verror)
        {
                if (verror < m_verror)
                {
                        m_params = params;
                        m_tvalue = tvalue;
                        m_terror = terror;
                        m_vvalue = vvalue;
                        m_verror = verror;
                        return true;
                }

                else
                {
                        return false;
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool trainer_state_t::update(const trainer_state_t& state)
        {
                return update(state.m_params, state.m_tvalue, state.m_terror, state.m_vvalue, state.m_verror);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        trainer_data_t::trainer_data_t(type t)
                :       m_type(t)
        {
                clear();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        trainer_data_t::trainer_data_t(const model_t& model, type t)
                :       m_type(t)
        {
                clear(model);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::clear(const model_t& model)
        {
                m_model = model.clone();
                m_vgrad.resize(m_model->n_parameters());

                clear();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::clear()
        {
                m_value = 0.0;
                m_error = 0.0;
                m_count = 0;
                m_vgrad.setZero();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::clear(const vector_t& x)
        {
                assert(m_model);

                m_model->load_params(x);
                clear();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::clear(type t)
        {
                m_type = t;
                clear();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::update(const task_t& task, const sample_t& sample, const loss_t& loss)
        {
                assert(m_model);

                const image_t& image = task.image(sample.m_index);
                const vector_t target = image.make_target(sample.m_region);
                assert(image.has_target(target));

                const vector_t output = m_model->value(image, sample.m_region);
                if (m_type == type::vgrad)
                {
                        m_vgrad.noalias() += m_model->gradient(loss.vgrad(target, output));
                }

                m_value += loss.value(target, output);
                m_error += loss.error(target, output);
                m_count ++;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::update(const tensor3d_t& input, const vector_t& target, const loss_t& loss)
        {
                assert(m_model);

                const vector_t output = m_model->value(input);
                if (m_type == type::vgrad)
                {
                        m_vgrad.noalias() += m_model->gradient(loss.vgrad(target, output));
                }

                m_value += loss.value(target, output);
                m_error += loss.error(target, output);
                m_count ++;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::update_st(const task_t& task, const samples_t& samples, const loss_t& loss)
        {
                for (size_t i = 0; i < samples.size(); i ++)
                {
                        update(task, samples[i], loss);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void trainer_data_t::update_mt(const task_t& task, const samples_t& samples, const loss_t& loss, size_t nthreads)
        {
                thread_loop_cumulate<trainer_data_t>
                (
                        samples.size(),
                        [&] (trainer_data_t& data)
                        {
                                assert(m_model);
                                data.clear(m_type);
                                data.clear(*m_model);
                        },
                        [&] (size_t i, trainer_data_t& data)
                        {
                                data.update(task, samples[i], loss);
                        },
                        [&] (trainer_data_t& data)
                        {
                                this->operator +=(data);
                        },
                        nthreads
                );
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        trainer_data_t& trainer_data_t::operator+=(const trainer_data_t& other)
        {
                m_value += other.m_value;
                m_error += other.m_error;
                m_vgrad += other.m_vgrad;
                m_count += other.m_count;
                return *this;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
