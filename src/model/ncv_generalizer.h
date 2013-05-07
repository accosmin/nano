#ifndef NANOCV_GENERALIZER_H
#define NANOCV_GENERALIZER_H

//#include "ncv_ml.h"

//namespace ncv
//{
//	////////////////////////////////////////////////////////////////////////////////
//	// Selects from a set of models the one that generalizes
//	//	the best on the validation dataset.
//	////////////////////////////////////////////////////////////////////////////////
	
//	template <class TModel>
//        class generalizer
//	{
//	public:
		
//		// Constructor
//                generalizer()
//                        :	m_train_error(0.0), m_valid_error(0.0)
//		{
//		}
                
//                // Clear history
//                void clear()
//                {
//                        m_train_errors.clear();
//                        m_valid_errors.clear();
//                }
		
//		// Process a new model with the given training and validation loss values
//		//	(discard <model> if it is overfitting)
//		bool process(scalar_t train_error, scalar_t valid_error,
//			     const TModel& model, const string_t& description)
//		{
//                        bool better = false;
                        
//			// Is <model> the best so far?!
//			if (m_valid_errors.empty() ||
//			    valid_error < *std::min_element(m_valid_errors.begin(), m_valid_errors.end()))
//			{
//				m_model = model;
//				m_description = description;
//				m_train_error = train_error;
//				m_valid_error = valid_error;
//                                better = true;
//			}
			
//			m_train_errors.push_back(train_error);
//			m_valid_errors.push_back(valid_error);
//                        m_descriptions.push_back(description);
                        
//                        return better;
//		}
		
//		// Access functions
//		const TModel& model() const { return m_model; }
//		const string_t& description() const { return m_description; }
//		scalar_t train_error() const { return m_train_error; }
//		scalar_t valid_error() const { return m_valid_error; }
//                scalar_t last_train_error() const { return last(m_train_errors); }
//                scalar_t last_valid_error() const { return last(m_valid_errors); }
//                string_t last_description() const { return last(m_descriptions); }
		
//	private:
		
//		// Returns the last entry in a list of values (if any)
//		template <class T>
//		static T last(const std::vector<T>& values)
//		{
//			return values.empty() ? T() : *values.rbegin();
//		}
			
//		// Attributes
//		scalars_t	m_train_errors;
//		scalars_t	m_valid_errors;
//                strings_t       m_descriptions;
                
//		string_t	m_description;	// Description of the optimal parameters
//		scalar_t	m_train_error;	// Optimal train loss
//		scalar_t	m_valid_error;	// Optimal validation loss
//		TModel		m_model;	// Optimal model
//	};
//}

#endif // NANOCV_GENERALIZER_H

