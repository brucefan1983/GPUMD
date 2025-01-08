.. _kw_output_descriptor:
.. index::
   single: output_descriptor (keyword in nep.in)

:attr:`output_descriptor`
=========================

This keyword instructs :program:`nep` to output the (normalized) descriptors for all the atoms in the ``train.xyz`` file.
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
