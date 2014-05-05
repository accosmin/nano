#ifndef NANOCV_CLONABLE_H
#define NANOCV_CLONABLE_H

#include <string>
#include <memory>

namespace ncv
{
        ///
        /// the clonable interface to be used with a manager:
        ///      ::clone(const std::string&)    - create a new object (with the given configuration)
        ///      ::configuration()              - parametrization
        ///      ::description()                - short description (configuration included)
        ///
        template
        <
                typename tobject
        >
        class clonable_t
        {        
        public:

                typedef std::shared_ptr<tobject> robject_t;

                ///
                /// \brief constructor
                ///
                clonable_t(const std::string& configuration,
                           const std::string& description)
                        :       m_configuration(configuration),
                                m_description(description)
                {
                }

                // disable copying
                clonable_t(const clonable_t&) = delete;
                clonable_t& operator=(const clonable_t&) = delete;

                ///
                /// \brief create an object clone
                ///
                robject_t clone() const { return clone(configuration()); }
                virtual robject_t clone(const std::string& params) const = 0;
                
                ///
                /// \brief describe the object
                ///
                const std::string& configuration() const { return m_configuration; }
                const std::string& description() const { return m_description; }

        protected:

                // attributes
                std::string     m_configuration;
                std::string     m_description;
        };
}

#endif // NANOCV_CLONABLE_H

