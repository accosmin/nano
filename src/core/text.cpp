#include "text.h"
#include <boost/algorithm/string.hpp>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        string_t text::resize(const string_t& str, size_t str_size, align alignment)
	{
                const size_t fill_size = str.size() > str_size ? 0 : str_size - str.size();

		switch (alignment)
		{
                case align::center:
			return string_t(fill_size / 2, ' ') + str + string_t(fill_size - fill_size / 2, ' ');

                case align::right:
			return string_t(fill_size, ' ') + str;

                case align::left:
		default:
			return str + string_t(fill_size, ' ');
		}
        }

	/////////////////////////////////////////////////////////////////////////////////////////
}
