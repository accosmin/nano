#ifndef NANOCV_MODEL_H
#define NANOCV_MODEL_H

//#include "ncv_msize.h"
//#include "ncv_clonable.h"

//namespace ncv
//{
//        class model_t;
//        typedef typename clonable_t<model_t>::robject_t         rmodel_t;

//        class dataset_t;
//        class loss_t;
	
//        /////////////////////////////////////////////////////////////////////////////////////////
//        // Model.
//        /////////////////////////////////////////////////////////////////////////////////////////
                
//        class model_t : public msize_t, public clonable_t<model_t>
//        {
//        public:

//                // Constructor
//                model_t(size_t rows = 0, size_t cols = 0, size_t outputs = 0)
//                        :       msize_t(rows, cols, outputs) {}

//                // Destructor
//                virtual ~model_t() {}
                
//                // Train the model
//                virtual bool train(const dataset_t& tdata, const dataset_t& vdata, const loss_t& loss,
//                        scalar_t eps, size_t iterations) = 0;
		
//		// Compute the model output
//                virtual const vector_t& process(const vector_t& input) = 0;
//                virtual const vector_t& process(const matrix_t& image, int x, int y) = 0;
                
//                // Save/load from file
//                virtual bool save(const string_t& path) const = 0;
//                virtual bool load(const string_t& path) = 0;
//        };
//}

#endif // NANOCV_MODEL_H
