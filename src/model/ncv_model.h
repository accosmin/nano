#ifndef NANOCV_MODEL_H
#define NANOCV_MODEL_H

#include "ncv_task.h"
#include "ncv_loss.h"

namespace ncv
{
        // manage models (register new ones, query and clone them)
        class model_t;
        typedef manager_t<model_t>              model_manager_t;
        typedef model_manager_t::robject_t      rmodel_t;

        /////////////////////////////////////////////////////////////////////////////////////////
        // generic model: output = model(input).
        /////////////////////////////////////////////////////////////////////////////////////////
                
        class model_t : public clonable_t<model_t>
        {
        public:

                // constructor
                model_t(const string_t& name, const string_t& description);

                // destructor
                virtual ~model_t() {}

                // train the model
                bool train(const task_t& task, const fold_t& fold, const loss_t& loss);

                // evaluate the model (compute the average loss value & error)
                void test(const task_t& task, const fold_t& fold, const loss_t& loss,
                        scalar_t& lvalue, scalar_t& lerror) const;

                // compute the model output
                virtual vector_t process(const image_t& image, coord_t x, coord_t y) const = 0;
                vector_t process(const image_t& image, const rect_t& region) const;

                // save/load from file
                bool save(const string_t& path) const;
                bool load(const string_t& path);

                // access functions
                size_t n_rows() const { return m_rows; }
                size_t n_cols() const { return m_cols; }
                size_t n_outputs() const { return m_outputs; }
                size_t n_parameters() const { return m_parameters; }

        protected:

                // apply an operator to each sample
                template
                <
                        typename toperator
                >
                void foreach_sample_with_target(const task_t& task, const samples_t& samples, toperator op) const
                {
                        for (size_t i = 0; i < samples.size(); i ++)
                        {
                                const sample_t& sample = samples[i];
                                const image_t& image = task.image(sample.m_index);

                                const vector_t target = image.get_target(sample.m_region);
                                if (image.has_target(target))
                                {
                                        const vector_t output = process(image, sample.m_region);
                                        op(i, output, target);
                                }
                        }
                }

                // save/load from file
                virtual bool save(std::ofstream& os) const = 0;
                virtual bool load(std::ifstream& is) = 0;

                // resize to new inputs/outputs, returns the number of parameters
                virtual size_t resize() = 0;

                // initialize parameters
                virtual void zero() = 0;
                virtual void random() = 0;

                // train the model
                virtual bool train(const task_t& task, const samples_t& samples, const loss_t& loss) = 0;

        protected:

                // initialize matrices & vectors
                static void zero(matrix_t& mat);
                static void zero(matrices_t& mats);
                static void zero(vector_t& vec);

                static void random(scalar_t min, scalar_t max, matrix_t& mat);
                static void random(scalar_t min, scalar_t max, matrices_t& mats);
                static void random(scalar_t min, scalar_t max, vector_t& vec);

                // serialize/deserialize matrices & vectors
                static void serialize(const matrix_t& mat, size_t& pos, vector_t& params);
                static void serialize(const matrices_t& mats, size_t& pos, vector_t& params);
                static void serialize(const vector_t& vec, size_t& pos, vector_t& params);

                static void deserialize(matrix_t& mat, size_t& pos, const vector_t& params);
                static void deserialize(matrices_t& mats, size_t& pos, const vector_t& params);
                static void deserialize(vector_t& vec, size_t& pos, const vector_t& params);

        private:

                // attributes
                size_t          m_rows, m_cols;         // input patch size
                size_t          m_outputs;              // output size
                size_t          m_parameters;
        };
}

#endif // NANOCV_MODEL_H
