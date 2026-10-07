##############################################################################
#
# Copyright (c) 2026 Zope Foundation and Contributors.
# All Rights Reserved.
#
# This software is subject to the provisions of the Zope Public License,
# Version 2.1 (ZPL).  A copy of the ZPL should accompany this distribution.
# THIS SOFTWARE IS PROVIDED "AS IS" AND ANY AND ALL EXPRESS OR IMPLIED
# WARRANTIES ARE DISCLAIMED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY, TITLE, AND FITNESS FOR A PARTICULAR
# PURPOSE.
#
##############################################################################
"""Regression test for a reference leak in the lazy declarations import
done by the C optimizations (see issue #363).

For Python 3.11+, ``_zic_state_load_declarations`` imports
``zope.interface.declarations`` and pulls a handful of attributes off of it
the first time a module-level function such as ``getObjectSpecification``
needs them. Every early return on that path used to drop the module (and
any attribute already fetched) without decrefing it first. The failure is
only reached when ``PyImport_ImportModule`` succeeds but a later
``PyObject_GetAttrString`` does not, which does not happen in ordinary
use -- so the reproducer below breaks the target attribute on purpose and
loads a second, independent instance of the extension module so the
module-level "already imported" cache does not just short circuit the call.

Note: The static-types path (Python < 3.11) is not covered by a test.
Sharing the _run_child helper with test_lookup_concurrency.py is deferred.
"""
import os
import subprocess
import sys
import textwrap
import unittest


# Subprocess harness.  A test runner that injects egg paths into sys.path
# in process -- buildout's bin/test via zc.recipe.testrunner -- leaves
# nothing for a subprocess to inherit, and the child then dies on the import
# instead of running the reproducer.  Passing them on through PYTHONPATH is a
# workaround, not an exact copy of this process's import state: it cannot
# carry an entry containing os.pathsep, it places these paths ahead of the
# child's standard library, and it does not reproduce import hooks.  Good
# enough to reach the reproducer, which is all the test needs.
def _make_subprocess_env():
    """Create a clean PYTHONPATH environment for subprocess calls."""
    return dict(os.environ, PYTHONPATH=os.pathsep.join(
        p for p in sys.path if p))


# Attributes in order of access by _zic_state_load_declarations
_ATTRS = [
    "BuiltinImplementationSpecifications",
    "_empty",
    "implementedByFallback",
    "Implements",
]


def _var_name(attr):
    """Generate a clean variable name for an attribute.

    Strips leading underscores and uses shorter names for common attributes.
    """
    # Short names for common attributes
    short_names = {
        'BuiltinImplementationSpecifications': 'builtin',
        'implementedByFallback': 'fallback',
        'Implements': 'implements',
        '_empty': 'empty',
    }
    return short_names.get(attr, attr.lstrip('_'))


def _make_child_code(break_index, break_value=None):
    """Generate child code that breaks attribute at break_index.

    Args:
        break_index: index into _ATTRS for which attribute to break
        break_value: if provided, set attribute to this value instead of
            deleting
    """
    # Build the code lines
    lines = []

    # Header
    lines.append('    import gc')
    lines.append('    import importlib.util')
    lines.append('    import sys')
    lines.append('')
    lines.append('    import zope.interface.declarations as decl_mod')
    lines.append('')
    lines.append('    spec = importlib.util.find_spec(')
    lines.append('        "zope.interface._zope_interface_coptimizations")')
    lines.append('    if spec is None or spec.loader is None:')
    lines.append('        raise SystemExit(')
    lines.append('            "could not locate the C extension module spec")')
    lines.append('')
    lines.append('    def fresh_module():')
    lines.append('        module = importlib.util.module_from_spec(spec)')
    lines.append('        spec.loader.exec_module(module)')
    lines.append('        return module')
    lines.append('')
    lines.append('    class Foo:')
    lines.append('        pass')
    lines.append('')

    # Save attributes up to and including break_index
    saved = []
    for i in range(break_index + 1):
        a = _ATTRS[i]
        var = _var_name(a)
        lines.append(f'    saved_{var} = decl_mod.{a}')
        saved.append((a, var))
    lines.append('')

    # Break the attribute at break_index
    attr = _ATTRS[break_index]
    if break_value is not None:
        lines.append(f'    decl_mod.{attr} = {break_value}')
    else:
        lines.append(f'    del decl_mod.{attr}')
    lines.append('')

    # Try block
    lines.append('    try:')
    lines.append('        broken = fresh_module()')
    lines.append('        gc.collect()')

    # Before refcounts for all saved attrs + module
    lines.append('        before = {}')
    for a, var in saved:
        lines.append(f'        before_{var} = sys.getrefcount(saved_{var})')
    lines.append('        before_decl_mod = sys.getrefcount(decl_mod)')
    lines.append('')

    # Call that should fail
    lines.append('        try:')
    if break_value is not None:
        lines.append('            broken.getObjectSpecification(Foo())')
        lines.append('        except TypeError as e:')
        lines.append('            if "not a type" not in str(e):')
        lines.append('                raise')
    else:
        lines.append('            broken.getObjectSpecification(Foo())')
        lines.append('        except AttributeError:')
        lines.append('            pass')
    lines.append('        else:')
    lines.append('            raise SystemExit(')
    lines.append(
        '                "expected error, call unexpectedly succeeded")')
    lines.append('')

    # After refcounts
    lines.append('        gc.collect()')
    for a, var in saved:
        lines.append(f'        after_{var} = sys.getrefcount(saved_{var})')
    lines.append('        after_decl_mod = sys.getrefcount(decl_mod)')
    lines.append('')

    # Checks
    for a, var in saved:
        a_short = ('BuiltinImplSpec'
                   if a == 'BuiltinImplementationSpecifications' else a)
        lines.append(f'        if after_{var} != before_{var}:')
        lines.append('            raise SystemExit(')
        lines.append(
            '                "%s refcount went from %%d to %%d "' %
            a_short)
        lines.append(
            f'                "across a failed import" %\n'
            f'                (before_{var}, after_{var}))')
    lines.append('        if after_decl_mod != before_decl_mod:')
    lines.append('            raise SystemExit(')
    lines.append('                "declarations module refcount went from %d' +
                 ' to %d "')
    lines.append('                "across a failed import" %')
    lines.append('                (before_decl_mod, after_decl_mod))')
    lines.append('')

    # Finally - restore
    lines.append('    finally:')
    for a, var in saved:
        lines.append(f'        decl_mod.{a} = saved_{var}')
    lines.append('')

    # Recovery test on same instance
    lines.append('    broken.getObjectSpecification(Foo())')
    lines.append('    print("ok")')

    return textwrap.dedent('\n'.join(lines))


# Generate child codes
# Break BuiltinImplementationSpecifications
_CHILD_FIRST_ATTR = _make_child_code(0)
_CHILD_SECOND_ATTR = _make_child_code(1)  # Break _empty
_CHILD_THIRD_ATTR = _make_child_code(2)  # Break implementedByFallback
_CHILD_IMPLEMENTS_NOT_TYPE = _make_child_code(
    3, break_value=42)  # Set Implements to non-type


class DeclarationsImportRefcountTests(unittest.TestCase):

    def _requires_c_extension(self):
        """Check if the C extension is available."""
        from zope.interface.adapter import LookupBase
        return LookupBase.__module__ == "_zope_interface_coptimizations"

    def _requires_heap_types(self):
        """Check if we're on Python 3.11+ with heap types."""
        return sys.version_info >= (3, 11)

    def _run_child(self, code):
        """Run Python code in a subprocess and return the result."""
        env = _make_subprocess_env()
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        return result

    def test_failed_first_attr_does_not_leak(self):
        """Test that failure on first attribute does not leak module ref."""
        if not self._requires_c_extension():
            self.skipTest("requires the C implementation")
        if not self._requires_heap_types():
            self.skipTest("requires Python 3.11+")

        result = self._run_child(_CHILD_FIRST_ATTR)
        self.assertEqual(
            result.returncode, 0,
            "first attr check failed (returncode %r):\n%s"
            % (result.returncode, result.stderr[-2000:]))
        self.assertIn("ok", result.stdout)

    def test_failed_second_attr_does_not_leak(self):
        """Test that failure on second attribute does not leak first attr."""
        if not self._requires_c_extension():
            self.skipTest("requires the C implementation")
        if not self._requires_heap_types():
            self.skipTest("requires Python 3.11+")

        result = self._run_child(_CHILD_SECOND_ATTR)
        self.assertEqual(
            result.returncode, 0,
            "second attr check failed (returncode %r):\n%s"
            % (result.returncode, result.stderr[-2000:]))
        self.assertIn("ok", result.stdout)

    def test_failed_third_attr_does_not_leak(self):
        """Test that failure on third attribute does not leak first two."""
        if not self._requires_c_extension():
            self.skipTest("requires the C implementation")
        if not self._requires_heap_types():
            self.skipTest("requires Python 3.11+")

        result = self._run_child(_CHILD_THIRD_ATTR)
        self.assertEqual(
            result.returncode, 0,
            "third attr check failed (returncode %r):\n%s"
            % (result.returncode, result.stderr[-2000:]))
        self.assertIn("ok", result.stdout)

    def test_failed_implements_type_check_does_not_leak(self):
        """Test Implements type check failure does not leak attributes."""
        if not self._requires_c_extension():
            self.skipTest("requires the C implementation")
        if not self._requires_heap_types():
            self.skipTest("requires Python 3.11+")

        result = self._run_child(_CHILD_IMPLEMENTS_NOT_TYPE)
        self.assertEqual(
            result.returncode, 0,
            "implements type check failed (returncode %r):\n%s"
            % (result.returncode, result.stderr[-2000:]))
        self.assertIn("ok", result.stdout)


def test_suite():
    return unittest.defaultTestLoader.loadTestsFromName(__name__)
