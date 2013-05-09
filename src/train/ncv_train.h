#ifndef NANOCV_TRAIN_H
#define NANOCV_TRAIN_H

#include "ncv_model.h"
#include "ncv_task.h"

namespace ncv
{	
        class loss_t;

        // evaluate a model on a given task: compute the average loss value & error
        void evaluate(const model_t& model, const loss_t& loss,
                      const task_t& task, const fold_t& fold,
                      scalar_t& lvalue, scalar_t& lerror);


        // TODO: generic trainer?! (given model, task and loss)

//        /////////////////////////////////////////////////////////////////////////////////////////
//        // Linear model:
//        //	output = weights * filter(input) + bias.
//        /////////////////////////////////////////////////////////////////////////////////////////
                
//        class linear_model_t : public model_t
//        {
//        public:
                
//                // Constructor
//                linear_model_t(size_t rows = 0, size_t cols = 0, size_t outputs = 0);

//                // Clone the object
//                virtual rmodel_t clone() const { return rmodel_t(new linear_model_t(*this)); }
		
//		// Resize
//                virtual void resize(size_t rows, size_t cols, size_t outputs);
//                virtual void resize(const msize_t& msize);
                
//                // Train the model
//                virtual bool train(const dataset_t& tdata, const dataset_t& vdata, const loss_t& loss,
//                        scalar_t eps, size_t iterations);
		
//		// Compute the model output
//                virtual const vector_t& process(const vector_t& input);
//                virtual const vector_t& process(const matrix_t& image, int x, int y);
                
//                // Save/load from file
//                virtual bool save(const string_t& path) const;
//                virtual bool load(const string_t& path);
		
//	private:
		
//		// Optimize parameters
//		bool optimize(const dataset_t& tdata, const dataset_t& vdata, const loss_t& loss,
//                        scalar_t eps, size_t iterations);
                
//        private:
                
//                // Attributes
//                matrix_t		m_weights;	// Parameters: weights
//		vector_t		m_bias;		// Parameters: bias
//                vector_t		m_input;	// (Processed) input
//		vector_t 		m_output;	// Output
//        };
}

#endif // NANOCV_TRAIN_H
