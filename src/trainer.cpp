#include "trainer.h"
#include "core/thread/thread.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace impl
        {
                static scalar_t lvalue(const task_t& task, const sample_t& sample, const loss_t& loss, const model_t& model)
                {
                        const image_t& image = task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);
                        assert(image.has_target(target));

                        const vector_t output = model.value(image, sample.m_region);
                        return loss.value(target, output);
                }

                static scalar_t lvgrad(const task_t& task, const sample_t& sample, const loss_t& loss, const model_t& model)
                {
                        const image_t& image = task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);
                        assert(image.has_target(target));

                        const vector_t output = model.value(image, sample.m_region);
                        model.cumulate_grad(loss.vgrad(target, output));
                        return loss.value(target, output);
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        struct value_data_t
        {
                // constructor
                value_data_t() : m_value(0.0), m_count(0)
                {
                }

                // update the cumulated loss value with a new sample
                void update(const task_t& task, const sample_t& sample, const loss_t& loss)
                {
                        update(task, sample, loss, *m_model);
                }
                void update(const task_t& task, const sample_t& sample, const loss_t& loss, const model_t& model)
                {
                        m_value += impl::lvalue(task, sample, loss, model);
                        m_count ++;
                }

                // cumulate loss value
                value_data_t& operator+=(const value_data_t& other)
                {
                        m_value += other.m_value;
                        m_count += other.m_count;
                        return *this;
                }

                scalar_t value() const { return m_value / ((m_count == 0) ? 1.0 : m_count); }

                // attributes
                scalar_t        m_value;
                size_t          m_count;
                rmodel_t        m_model;
        };

        /////////////////////////////////////////////////////////////////////////////////////////

        struct vgrad_data_t
        {
                // constructor
                vgrad_data_t(size_t n_parameters = 0) : m_value(0.0), m_count(0)
                {
                        resize(n_parameters);
                }

                void resize(size_t n_parameters)
                {
                        m_vgrad.resize(n_parameters);
                        m_vgrad.setZero();
                }

                // update the cumulated loss value & gradient with a new sample
                void update(const task_t& task, const sample_t& sample, const loss_t& loss)
                {
                        update(task, sample, loss, *m_model);
                }
                void update(const task_t& task, const sample_t& sample, const loss_t& loss, const model_t& model)
                {
                        m_value += impl::lvgrad(task, sample, loss, model);
                        m_count ++;
                }
                void store() const
                {
                        m_vgrad = m_model->grad();
                }
                void store(const model_t& model) const
                {
                        m_vgrad = model.grad();
                }

                // cumulate loss value & gradient
                vgrad_data_t& operator+=(const vgrad_data_t& other)
                {
                        m_value += other.m_value;
                        m_vgrad += other.m_vgrad;
                        m_count += other.m_count;
                        return *this;
                }

                scalar_t value() const { return m_value / ((m_count == 0) ? 1.0 : m_count); }
                vector_t vgrad() const { return m_vgrad / ((m_count == 0) ? 1.0 : m_count); }

                // attributes
                scalar_t                m_value;
                mutable vector_t        m_vgrad;
                size_t                  m_count;
                rmodel_t                m_model;
        };

        /////////////////////////////////////////////////////////////////////////////////////////

        samples_t prune_annotated(const task_t& task, const samples_t& samples)
        {
                samples_t pruned_samples;

                // keep only the samples having targets associated
                for (const sample_t& sample : samples)
                {
                        const image_t& image = task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);
                        if (image.has_target(target))
                        {
                                pruned_samples.push_back(sample);
                        }
                }

                return pruned_samples;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t lvalue(const task_t& task, const sample_t& sample, const loss_t& loss, const model_t& model)
        {
                return impl::lvalue(task, sample, loss, model);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t lvgrad(const task_t& task, const sample_t& sample, const loss_t& loss, const model_t& model)
        {
                return impl::lvgrad(task, sample, loss, model);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t lvalue_st(const task_t& task, const samples_t& samples, const loss_t& loss, const model_t& model)
        {
                value_data_t cum_data;

                for (size_t i = 0; i < samples.size(); i ++)
                {
                        cum_data.update(task, samples[i], loss, model);
                }

                return cum_data.value();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t lvalue_mt(
                const task_t& task, const samples_t& samples, const loss_t& loss, size_t nthreads, const model_t& model)
        {
                value_data_t cum_data;

                thread_loop_cumulate<value_data_t>
                (
                        samples.size(),
                        [&] (value_data_t& data)
                        {
                                data.m_model = model.clone();
                        },
                        [&] (size_t i, value_data_t& data)
                        {
                                data.update(task, samples[i], loss);
                        },
                        [&] (const value_data_t& data)
                        {
                                cum_data += data;
                        },
                        nthreads
                );

                return cum_data.value();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t lvgrad_st(const task_t& task, const samples_t& samples, const loss_t& loss, const model_t& model,
                vector_t& lgrad)
        {
                vgrad_data_t cum_data(model.n_parameters());

                model.zero_grad();

                for (size_t i = 0; i < samples.size(); i ++)
                {
                        cum_data.update(task, samples[i], loss, model);
                }

                cum_data.store(model);

                lgrad = cum_data.vgrad();
                return cum_data.value();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t lvgrad_mt(
                const task_t& task, const samples_t& samples, const loss_t& loss, size_t nthreads,
                const model_t& model, vector_t& lgrad)
        {
                vgrad_data_t cum_data(model.n_parameters());

                thread_loop_cumulate<vgrad_data_t>
                (
                        samples.size(),
                        [&] (vgrad_data_t& data)
                        {
                                data.resize(model.n_parameters());
                                data.m_model = model.clone();
                                data.m_model->zero_grad();
                        },
                        [&] (size_t i, vgrad_data_t& data)
                        {
                                data.update(task, samples[i], loss);
                        },
                        [&] (const vgrad_data_t& data)
                        {
                                data.store();
                                cum_data += data;
                        },
                        nthreads
                );

                lgrad = cum_data.vgrad();
                return cum_data.value();
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
	
