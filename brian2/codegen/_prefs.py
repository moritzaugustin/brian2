'''
Module declaring general code generation preferences.
'''

from .codeobject import CodeObject
from brian2.core.preferences import prefs, BrianPreference

# Preferences
prefs.register_preferences(
    'codegen',
    'Code generation preferences',
    target=BrianPreference(
        default='numpy',
        docs='''
        Default target for code generation.
        
        Can be a string, in which case it should be one of:
        
        * ``'numpy'`` by default because this works on all platforms, but may not
          be maximally efficient.
        * `'weave'` uses ``scipy.weave`` to generate and compile C++ code,
          should work anywhere where ``gcc`` is installed and available at the
          command line.
        * ``'cython'``, uses the Cython package to generate C++ code. Needs a
          working installation of Cython and a C++ compiler.
        
        Or it can be a ``CodeObject`` class.
        ''',
        validator=lambda target: isinstance(target, basestring) or issubclass(target, CodeObject),
        ),
    string_expression_target=BrianPreference(
        default='numpy',
        docs='''
        Default target for the evaluation of string expressions (e.g. when
        indexing state variables). Should normally not be changed from the
        default numpy target, because the overhead of compiling code is not
        worth the speed gain for simple expressions.

        Accepts the same arguments as `codegen.target`.
        ''',
        validator=lambda target: isinstance(target, basestring) or issubclass(target, CodeObject),
    )
    )
