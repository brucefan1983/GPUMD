.. _kw_output_descriptor:
.. index::
   single: output_descriptor (keyword in nep.in)

:attr:`output_descriptor`
=========================

Write normalized descriptors for the structures or atoms in ``train.xyz`` to :ref:`descriptor.out <descriptor_out>`.
It is only in effect for the prediction mode (see the :ref:`prediction keyword <kw_prediction>`).
The syntax is::

  output_descriptor <mode>

where :attr:`<mode>` must be an integer that can assume one of the following values.

=====  =====================================================
Value  Mode 
-----  -----------------------------------------------------
0      not to output descriptors during prediction (default)
1      output per-structure descriptors during prediction
2      output per-atom descriptors during prediction
=====  =====================================================
