.. _kw_compute_orientorder:
.. index::
   single: compute_orientorder (keyword in run.in)

:attr:`compute_orientorder`
============================

This keyword computes the local, rotationally invariant Steinhardt order parameters 
:math:`q_l` and :math:`w_l`, as described in [Steinhardt1983]_ and [Mickel2013]_.  
The results are written to the :ref:`orientorder.out <orientorder_out>` file.

For an atom :math:`i`, the parameter :math:`q_l(i)` is defined as:

.. math::
   
   q_l(i) = \sqrt{\frac{4\pi}{2l+1}\sum_{m=-l}^{l}{|q_{lm}(i)|}^2}

where the quantity :math:`q_{lm}(i)` is obtained by summing the spherical harmonics 
over atom :math:`i` and its neighbors :math:`j`:

.. math::

    q_{lm}(i)=\frac{1}{N_b}\sum_{j=1}^{N_b}Y_{lm}(\theta({\vec{r}_{ij}}), \phi({\vec{r}_{ij}})).

If ``wl`` is set to ``True``, the quantity :math:`w_l` is computed. This is a weighted 
average of the :math:`q_{lm}(i)` values using `Wigner 3-j symbols <https://en.wikipedia.org/wiki/3-j_symbol>`_, 
yielding a **rotationally invariant combination**:

.. math::

   w_l(i) = \sum \limits_{m_1 + m_2 + m_3 = 0} \begin{pmatrix}
            l & l & l \\
            m_1 & m_2 & m_3
        \end{pmatrix}
        q_{lm_1}(i) q_{lm_2}(i) q_{lm_3}(i).

If the ``wl_hat`` parameter is ``True``, the :math:`w_l` order parameter is normalized as:

.. math::

   w_l(i) = \frac{
            \sum \limits_{m_1 + m_2 + m_3 = 0} \begin{pmatrix}
                l & l & l \\
                m_1 & m_2 & m_3
            \end{pmatrix}
            q_{lm_1}(i) q_{lm_2}(i) q_{lm_3}(i)}
            {\left(\sum \limits_{m=-l}^{l} |q_{lm}(i)|^2 \right)^{3/2}}.

If ``average`` is ``True``, the averaging procedure replaces :math:`q_{lm}(i)` 
with :math:`\overline{q}_{lm}(i)`, which is the mean value of :math:`q_{lm}(k)` over all 
neighbors :math:`k` of particle :math:`i`, including atom :math:`i` itself:

.. math::

   \overline{q}_{lm}(i) = \frac{1}{N_b} \sum \limits_{k=0}^{N_b}
   q_{lm}(k).


Syntax
------

.. code::

  compute_orientorder <interval> <mode_type> <mode_parameters> <ndegrees> <degree1> <degree2> ... <average> <wl> <wlhat>

- **interval**: perform the calculation every :attr:`interval` steps. 
- **mode_type**: the neighbor selection mode. Currently supports ``cutoff`` or ``nnn`` (nearest neighbor number).
- **mode_parameters**:  
  - for ``cutoff`` mode, this is the cutoff distance;  
  - for ``nnn`` mode, this is the number of neighbors.  
  **Note:** if an atom has fewer neighbors than ``nnn`` within 6 Å, the result is set to 0.
- **ndegrees**: number of spherical harmonic degrees (:math:`l`) to compute.
- **degree1..degreeN**: individual degree values.
- **average**: whether to calculate the averaged Steinhardt order parameter. ``1`` = True, ``0`` = False. Defaults to False.
- **wl**: whether to compute the :math:`w_l` version of the Steinhardt order parameter. Defaults to False.
- **wlhat**: whether to compute the **normalized** :math:`w_l` version. Defaults to False.


Examples
--------

- ``compute_orientorder 100 cutoff 4.0 3 4 6 8``  
  → Every 100 MD steps, compute three degrees (4, 6, and 8) using a 4 Å cutoff.

- ``compute_orientorder 50 nnn 12 2 4 6 0 1``  
  → Every 50 MD steps, compute two degrees (4 and 6) with 12 nearest neighbors.  
  Additionally, compute the :math:`w_l` version.

- ``compute_orientorder 100 nnn 12 2 4 6 1 1 1``  
  → Every 100 MD steps, compute two degrees (4 and 6) with 12 nearest neighbors.  
  Use the **averaged definition**, and calculate both the original and the normalized :math:`w_l` versions.
