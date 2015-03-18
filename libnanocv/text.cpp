#include "text.h"

namespace ncv
{
        std::string text::resize(const std::string& str, std::size_t str_size, align alignment, char fill_char)
	{
                const std::size_t fill_size = str.size() > str_size ? 0 : str_size - str.size();

		switch (alignment)
		{
                case align::center:
                        return std::string(fill_size / 2, fill_char) +
                               str +
                               std::string(fill_size - fill_size / 2, fill_char);

                case align::right:
                        return std::string(fill_size, fill_char) +
                               str;

                case align::left:
		default:
                        return str +
                               std::string(fill_size, fill_char);
		}
        }
}
