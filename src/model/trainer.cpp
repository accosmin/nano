#include "trainer.h"

namespace ncv
{
//        //-------------------------------------------------------------------------------------------------

//        struct value_data_t
//        {
//                value_data_t() : m_value(0.0), m_count(0) {}

//                scalar_t value() const { return ,+
//                {

//                }

//                scalar_t        m_value;
//                size_t          m_count;
//        };

        //-------------------------------------------------------------------------------------------------

        samples_t trainer_t::prune_annotated(const task_t& task, const samples_t& samples)
        {
                samples_t pruned_samples;

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

        //-------------------------------------------------------------------------------------------------

        scalar_t trainer_t::value(
                const task_t& task, const samples_t& samples, const loss_t& loss,
                const model_t& model)
        {
                scalar_t lvalue = 0.0;
                size_t lcount = 0;

                // TODO: parallelize this!
                //      clone the model, resize with the task, initialize with the given parameters
                //      and then update the loss

                for (const sample_t& sample : samples)
                {
                        const image_t& image = task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);
                        assert(image.has_target(target));

                        const vector_t output = model.value(image, sample.m_region);

                        lvalue += loss.value(target, output);
                        lcount ++;
                }

                lvalue /= (lcount == 0) ? 1.0 : lcount;

                return lvalue;
        }

        //-------------------------------------------------------------------------------------------------

        scalar_t trainer_t::vgrad(
                const task_t& task, const samples_t& samples, const loss_t& loss,
                const model_t& model, vector_t& lgrad)
        {
                scalar_t lvalue = 0.0;
                size_t lcount = 0;

                lgrad.resize(model.n_parameters());
                lgrad.setZero();

                // TODO: parallelize this!
                //      clone the model, resize with the task, initialize with the given parameters
                //      and then update the loss

                for (const sample_t& sample : samples)
                {
                        const image_t& image = task.image(sample.m_index);
                        const vector_t target = image.make_target(sample.m_region);
                        assert(image.has_target(target));

                        const vector_t output = model.value(image, sample.m_region);

                        lvalue += loss.value(target, output);
                        lcount ++;

                        const vector_t mgrad = model.vgrad(loss.vgrad(target, output));
                        assert(mgrad.size() == lgrad.size());
                        lgrad += mgrad;
                }

                lvalue /= (lcount == 0) ? 1.0 : lcount;
                lgrad /= (lcount == 0) ? 1.0 : lcount;

                return lvalue;
        }

        //-------------------------------------------------------------------------------------------------
}
