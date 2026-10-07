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


# Child code: break first attribute (BuiltinImplementationSpecifications)
_CHILD_FIRST_ATTR = textwrap.dedent("""
    import gc
    import importlib.util
    import sys

    import zope.interface.declarations as decl_mod

    spec = importlib.util.find_spec(
        "zope.interface._zope_interface_coptimizations")
    if spec is None or spec.loader is None:
        raise SystemExit("could not locate the C extension module spec")

    def fresh_module():
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    class Foo:
        pass

    saved = decl_mod.BuiltinImplementationSpecifications
    del decl_mod.BuiltinImplementationSpecifications
    try:
        broken = fresh_module()
        gc.collect()
        before = sys.getrefcount(decl_mod)
        try:
            broken.getObjectSpecification(Foo())
        except AttributeError:
            pass
        else:
            raise SystemExit(
                "expected AttributeError, call unexpectedly succeeded")
        gc.collect()
        after = sys.getrefcount(decl_mod)
        if after != before:
            raise SystemExit(
                "declarations module refcount went from %d to %d "
                "across a failed import" % (before, after))
    finally:
        decl_mod.BuiltinImplementationSpecifications = saved

    broken.getObjectSpecification(Foo())
    print("ok")
    """)


# Child code: break second attribute (_empty), check first attr refcount
_CHILD_SECOND_ATTR = textwrap.dedent("""
    import gc
    import importlib.util
    import sys

    import zope.interface.declarations as decl_mod

    spec = importlib.util.find_spec(
        "zope.interface._zope_interface_coptimizations")
    if spec is None or spec.loader is None:
        raise SystemExit("could not locate the C extension module spec")

    def fresh_module():
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    class Foo:
        pass

    saved_empty = decl_mod._empty
    saved_builtin = decl_mod.BuiltinImplementationSpecifications
    del decl_mod._empty
    try:
        broken = fresh_module()
        gc.collect()
        before_builtin = sys.getrefcount(saved_builtin)
        before = sys.getrefcount(decl_mod)
        try:
            broken.getObjectSpecification(Foo())
        except AttributeError:
            pass
        else:
            raise SystemExit(
                "expected AttributeError, call unexpectedly succeeded")
        gc.collect()
        after_builtin = sys.getrefcount(saved_builtin)
        after = sys.getrefcount(decl_mod)
        if after_builtin != before_builtin:
            raise SystemExit(
                "BuiltinImplSpec refcount went from %d to %d "
                "across a failed import" % (before_builtin, after_builtin))
        if after != before:
            raise SystemExit(
                "declarations module refcount went from %d to %d "
                "across a failed import" % (before, after))
    finally:
        decl_mod._empty = saved_empty

    broken.getObjectSpecification(Foo())
    print("ok")
    """)


# Child code: break third attribute (implementedByFallback)
_CHILD_THIRD_ATTR = textwrap.dedent("""
    import gc
    import importlib.util
    import sys

    import zope.interface.declarations as decl_mod

    spec = importlib.util.find_spec(
        "zope.interface._zope_interface_coptimizations")
    if spec is None or spec.loader is None:
        raise SystemExit("could not locate the C extension module spec")

    def fresh_module():
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    class Foo:
        pass

    saved_fallback = decl_mod.implementedByFallback
    saved_empty = decl_mod._empty
    saved_builtin = decl_mod.BuiltinImplementationSpecifications
    del decl_mod.implementedByFallback
    try:
        broken = fresh_module()
        gc.collect()
        before_builtin = sys.getrefcount(saved_builtin)
        before_empty = sys.getrefcount(saved_empty)
        before = sys.getrefcount(decl_mod)
        try:
            broken.getObjectSpecification(Foo())
        except AttributeError:
            pass
        else:
            raise SystemExit(
                "expected AttributeError, call unexpectedly succeeded")
        gc.collect()
        after_builtin = sys.getrefcount(saved_builtin)
        after_empty = sys.getrefcount(saved_empty)
        after = sys.getrefcount(decl_mod)
        if after_builtin != before_builtin:
            raise SystemExit(
                "BuiltinImplSpec refcount went from %d to %d "
                "across a failed import" % (before_builtin, after_builtin))
        if after_empty != before_empty:
            raise SystemExit(
                "_empty refcount went from %d to %d "
                "across a failed import" % (before_empty, after_empty))
        if after != before:
            raise SystemExit(
                "declarations module refcount went from %d to %d "
                "across a failed import" % (before, after))
    finally:
        decl_mod.implementedByFallback = saved_fallback

    broken.getObjectSpecification(Foo())
    print("ok")
    """)


# Child code: make Implements not a type
_CHILD_IMPLEMENTS_NOT_TYPE = textwrap.dedent("""
    import gc
    import importlib.util
    import sys

    import zope.interface.declarations as decl_mod

    spec = importlib.util.find_spec(
        "zope.interface._zope_interface_coptimizations")
    if spec is None or spec.loader is None:
        raise SystemExit("could not locate the C extension module spec")

    def fresh_module():
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    class Foo:
        pass

    saved_implements = decl_mod.Implements
    saved_fallback = decl_mod.implementedByFallback
    saved_empty = decl_mod._empty
    saved_builtin = decl_mod.BuiltinImplementationSpecifications
    decl_mod.Implements = 42  # Not a type
    try:
        broken = fresh_module()
        gc.collect()
        before_builtin = sys.getrefcount(saved_builtin)
        before_empty = sys.getrefcount(saved_empty)
        before_fallback = sys.getrefcount(saved_fallback)
        before = sys.getrefcount(decl_mod)
        try:
            broken.getObjectSpecification(Foo())
        except TypeError as e:
            if "not a type" not in str(e):
                raise
        else:
            raise SystemExit(
                "expected TypeError, call unexpectedly succeeded")
        gc.collect()
        after_builtin = sys.getrefcount(saved_builtin)
        after_empty = sys.getrefcount(saved_empty)
        after_fallback = sys.getrefcount(saved_fallback)
        after = sys.getrefcount(decl_mod)
        if after_builtin != before_builtin:
            raise SystemExit(
                "BuiltinImplSpec refcount went from %d to %d "
                "across a failed import" % (before_builtin, after_builtin))
        if after_empty != before_empty:
            raise SystemExit(
                "_empty refcount went from %d to %d "
                "across a failed import" % (before_empty, after_empty))
        if after_fallback != before_fallback:
            raise SystemExit(
                "implementedByFallback refcount went from %d to %d "
                "across failed import" % (before_fallback, after_fallback))
        if after != before:
            raise SystemExit(
                "declarations module refcount went from %d to %d "
                "across a failed import" % (before, after))
    finally:
        decl_mod.Implements = saved_implements

    broken.getObjectSpecification(Foo())
    print("ok")
    """)


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
