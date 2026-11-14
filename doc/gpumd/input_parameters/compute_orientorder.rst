.. _kw_compute_orientorder:
.. index::
   single: compute_orientorder (keyword in run.in)

:attr:`compute_orientorder`
============================

This keyword computes the local, rotationally invariant Steinhardt order parameters
:math:`q_l` and :math:`w_l`, as described in [Steinhardt1983]_ and [Mickel2013]_.

Syntax
------

::

  compute_orientorder <interval> <mode_type> <mode_parameter> <ndegrees> <degree1> ... <degreeN> [<average> [<wl> [<wlhat>]]]

Parameters
----------

* ``interval`` is the positive sampling interval in MD steps.
* ``mode_type`` selects the neighbors:

  * ``cutoff``: ``mode_parameter`` is a positive cutoff distance in Å.
  * ``nnn``: ``mode_parameter`` is a positive number of nearest neighbors.
    If fewer than this number are found within 6 Å, the result for that atom is zero.

* ``ndegrees`` is the positive number of spherical-harmonic degrees to calculate.
* ``degree1`` through ``degreeN`` are non-negative integer degrees.
* ``average`` enables the averaged definition.
* ``wl`` enables the unnormalized invariant :math:`w_l`.
* ``wlhat`` enables the normalized :math:`w_l`.

The three optional flags use 0 (disabled, the default) or 1 (enabled).
They are positional: supply earlier flags when setting a later one.

Output
------

The results are written to the :ref:`orientorder.out <orientorder_out>` file.

Examples
--------

Compute degrees 4, 6, and 8 every 100 steps using a 4 Å cutoff::

  compute_orientorder 100 cutoff 4.0 3 4 6 8

Compute degrees 4 and 6 and their unnormalized :math:`w_l` values using 12 nearest neighbors::

  compute_orientorder 50 nnn 12 2 4 6 0 1

Use the averaged definition and output both forms of :math:`w_l`::

  compute_orientorder 100 nnn 12 2 4 6 1 1 1

Notes
-----

Definitions
^^^^^^^^^^^

For an atom :math:`i`, the parameter :math:`q_l(i)` is defined as:

.. math::

   q_l(i) = \sqrt{\frac{4\pi}{2l+1}\sum_{m=-l}^{l}{|q_{lm}(i)|}^2}

where the quantity :math:`q_{lm}(i)` is obtained by summing the spherical harmonics
over the neighbors :math:`j` of atom :math:`i`:

.. math::

    q_{lm}(i)=\frac{1}{N_b}\sum_{j=1}^{N_b}Y_{lm}(\theta({\vec{r}_{ij}}), \phi({\vec{r}_{ij}})).

If ``wl`` is set to ``1``, the quantity :math:`w_l` is computed. This is a weighted
average of the :math:`q_{lm}(i)` values using `Wigner 3-j symbols <https://en.wikipedia.org/wiki/3-j_symbol>`_,
yielding a **rotationally invariant combination**:

.. math::

   w_l(i) = \sum \limits_{m_1 + m_2 + m_3 = 0} \begin{pmatrix}
            l & l & l \\
            m_1 & m_2 & m_3
        \end{pmatrix}
        q_{lm_1}(i) q_{lm_2}(i) q_{lm_3}(i).

If the ``wlhat`` parameter is ``1``, the :math:`w_l` order parameter is normalized as:

.. math::

   w_l(i) = \frac{
            \sum \limits_{m_1 + m_2 + m_3 = 0} \begin{pmatrix}
                l & l & l \\
                m_1 & m_2 & m_3
            \end{pmatrix}
            q_{lm_1}(i) q_{lm_2}(i) q_{lm_3}(i)}
            {\left(\sum \limits_{m=-l}^{l} |q_{lm}(i)|^2 \right)^{3/2}}.

If ``average`` is ``1``, the averaging procedure replaces :math:`q_{lm}(i)`
with :math:`\overline{q}_{lm}(i)`, which is the mean value of :math:`q_{lm}(k)` over all
neighbors :math:`k` of particle :math:`i`, including atom :math:`i` itself:

.. math::

   \overline{q}_{lm}(i) = \frac{1}{N_b+1} \sum \limits_{k=0}^{N_b}
   q_{lm}(k).
