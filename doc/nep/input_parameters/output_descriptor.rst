.. _kw_output_descriptor:
.. index::
   single: output_descriptor (keyword in nep.in)

:attr:`output_descriptor`
=========================

Select per-structure or per-atom descriptor output in prediction mode.

Syntax
------

::

  output_descriptor <mode>

Parameters
----------

:attr:`<mode>` must be an integer that can assume one of the following values.

=====  =====================================================
Value  Mode
-----  -----------------------------------------------------
0      not to output descriptors during prediction (default)
1      output per-structure descriptors during prediction
2      output per-atom descriptors during prediction
=====  =====================================================

Output
------

With ``prediction 1`` and mode 1 or 2, normalized descriptors from ``train.xyz`` are appended to :ref:`descriptor.out <descriptor_out>`.
Mode 1 writes one row per structure; mode 2 writes one row per atom.
No descriptor file is written during training.

Examples
--------

::

  prediction 1
  output_descriptor 1

Notes
-----

Descriptor output is not available for ``model nep_dipole`` or ``model nep_polarizability``.
