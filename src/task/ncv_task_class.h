#ifndef NANOCV_TASK_CLASSIFICATION_H
#define NANOCV_TASK_CLASSIFICATION_H

#include "ncv_task.h"

namespace ncv
{
//	////////////////////////////////////////////////////////////////////////////////
//        // classification task:
//        //      - all images have the same size (= the model size) and label
//        //      - the goal is to predict the correct label for a sample
//        //      - the sampling procedure selects uniformly distributed samples over their labels
//        //
//        // a sample consists of grayscale patches normalized to [-1, +1].
//	////////////////////////////////////////////////////////////////////////////////
	
//        class class_task : public task
//	{
//	public:
                
//                // Constructor
//                class_task_t(size_t rows, size_t cols, size_t outputs);
                
//                // Destructor
//                virtual ~class_task_t() {}

//                // Select <n_samples> to a dataset
//                virtual bool sample(data_enum dtype, size_t n_samples, dataset_t& dataset) const;

//                // Print a short description
//                virtual void describe() const;

//                // Test a model on this task
//                virtual bool test(const model_t& model, const loss_t& loss) const;
                
//                // Access functions
//                virtual size_t n_images() const { return _n_images(); }
//                virtual size_t n_images(data_enum dtype) const { return images(dtype).size(); }
//                virtual const image_t& image(data_enum dtype, index_t i) const { return images(dtype)[i]; }
                
//                virtual size_t n_samples() const { return n_images(); }
//                virtual size_t n_samples(data_enum dtype) const { return n_images(dtype); }
                
//        protected:

//                // Clear data
//                void clear();

//                // Set labels
//                void set_labels(const strings_t& labels);

//                // Add image
//                bool add_image(data_enum dtype, const image_t& image, index_t ilabel);

//                // Access functions
//                size_t n_labels() const { return m_labels.size(); }
//                const strings_t& labels() const { return m_labels; }
                
//        private:
                
//                // Total number of images
//                size_t _n_images() const
//                {
//                        return  n_images(data_enum::train) +
//                                n_images(data_enum::valid) +
//                                n_images(data_enum::test);
//                }

//                // Access functions
//                const images_t& images(data_enum dtype) const
//                {
//                        return m_images[cast<int>(dtype)];
//                }
//                const indices_t& ilabels(data_enum dtype) const
//                {
//                        return m_ilabels[cast<int>(dtype)];
//                }

//                images_t& images(data_enum dtype)
//                {
//                        return m_images[cast<int>(dtype)];
//                }
//                indices_t& ilabels(data_enum dtype)
//                {
//                        return m_ilabels[cast<int>(dtype)];
//                }

//                // Save a selected set of samples to dataset
//                void to_dataset(data_enum dtype, const indices_t& isamples, dataset_t& dataset) const;
		
//        private:
		
//		// Attributes
//                images_t                m_images[3];    // (train + valid + test) images
//                indices_t               m_ilabels[3];   // (train + valid + test) label indices

//                strings_t               m_labels;       // Distinct labels
//                vectors_t               m_targets;      // Distinct targets
//	};
}

#endif // NANOCV_TASK_CLASSIFICATION_H
