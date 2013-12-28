#ifndef NANOCV_TRAINER_DATA_H
#define NANOCV_TRAINER_DATA_H

#include "task.h"
#include "loss.h"
#include "model.h"
#include "thread/thread_loop.hpp"
#include <cassert>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // helper class to cumulate sample evaluations (loss value, error and gradient).
        /////////////////////////////////////////////////////////////////////////////////////////
                
        template
        <
                bool tgradient
        >
        class trainer_data_t
        {
        public:

                // constructor
                trainer_data_t()
                {
                        init();
                }

                trainer_data_t(const model_t& model)
                {
                        init(model);
                }

                // initialize statistics
                void init(const model_t& model)
                {
                        m_model = model.clone();
                        m_vgrad.resize(m_model->n_parameters());

                        init();
                }

                void init()
                {
                        m_value = 0.0;
                        m_error = 0.0;
                        m_count = 0;
                        m_vgrad.setZero();
                }

                // load parameters
                void load_params(const vector_t& x)
                {
                        assert(m_model);

                        m_model->load_params(x);
                        init();
                }

                // update statistics for a sample
                void update(const task_t& task, const sample_t& sample, const loss_t& loss)
                {
                        assert(m_model);

                        const image_t& image = task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);
                        assert(image.has_target(target));

                        const vector_t output = m_model->value(image, sample.m_region);
                        if (tgradient)
                        {
                                m_vgrad += m_model->gradient(loss.vgrad(target, output));
                        }

                        m_value += loss.value(target, output);
                        m_error += loss.error(target, output);
                        m_count ++;
                }

                // update statistics for a set of samples - single-threaded version
                void update_st(const task_t& task, const samples_t& samples, const loss_t& loss)
                {
                        for (size_t i = 0; i < samples.size(); i ++)
                        {
                                update(task, samples[i], loss);
                        }
                }

                // update statistics for a set of samples - multi-threaded version
                void update_mt(const task_t& task, const samples_t& samples, const loss_t& loss, size_t nthreads = 0)
                {
                        thread_loop_cumulate<trainer_data_t>
                        (
                                samples.size(),
                                [&] (trainer_data_t& data)
                                {
                                        assert(m_model);
                                        data.init(*m_model);
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

                // cumulate loss value & gradient
                trainer_data_t& operator+=(const trainer_data_t& other)
                {
                        m_value += other.m_value;
                        m_error += other.m_error;
                        m_vgrad += other.m_vgrad;
                        m_count += other.m_count;
                        return *this;
                }

                // access functions
                scalar_t value() const { return m_value / ((size() == 0) ? 1.0 : size()); }
                scalar_t error() const { return m_error / ((size() == 0) ? 1.0 : size()); }
                vector_t vgrad() const { return m_vgrad / ((size() == 0) ? 1.0 : size()); }
                size_t n_parameters() const { assert(m_model); return m_model->n_parameters(); }
                size_t size() const { return m_count; }

        private:

                // attributes
                scalar_t                m_value;
                scalar_t                m_error;
                vector_t                m_vgrad;
                size_t                  m_count;
                rmodel_t                m_model;
        };
}

#endif // NANOCV_TRAINER_DATA_H
