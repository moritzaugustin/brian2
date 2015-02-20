#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>
#include "run.h"
#include "rand.h"

{% for codeobj in code_objects %}
#include "code_objects/{{codeobj.name}}.h"
{% endfor %}

{% for name in additional_headers %}
#include "{{name}}"
{% endfor %}

#include <iostream>
#include <fstream>

{{report_func|autoindent}}

int main(int argc, char **argv)
{

	brian_start();

	{
		using namespace brian;

        {{main_lines|autoindent}}
	}

	brian_end();

	return 0;
}
