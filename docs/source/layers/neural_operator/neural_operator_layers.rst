.. _neural-operator-layers:

Neural Operator Layers
======================

* :ref:`neural_operator_layer_type <neural-operator-layer>` - Mean-field neural operator layer with local and global coupling
* :ref:`fixed_lno_layer_type <fixed-lno-layer>` - Laplace neural operator layer with fixed encoder/decoder bases and spectral mixing
* :ref:`dynamic_lno_layer_type <dynamic-lno-layer>` - Laplace neural operator layer with learnable encoder/decoder bases and spectral mixing
* :ref:`spectral_filter_layer_type <spectral-filter-layer>` - Spectral filter layer with fixed DCT bases and learnable per-mode weights
* :ref:`graph_nop_layer_type <graph-nop-layer>` - Graph neural operator layer with learned edge kernels on irregular graphs
* :ref:`orthogonal_nop_block_type <orthogonal-nop-block>` - Orthogonal neural operator block with learned orthonormal basis and spectral mixing
* :ref:`orthogonal_attention_layer_type <orthogonal-attention-layer>` - Orthogonal attention layer with low-rank basis projection and local bypass


.. toctree::
    :hidden:
    :maxdepth: 2

    neural_operator_layer
    fixed_lno_layer
    dynamic_lno_layer
    spectral_filter_layer
    graph_nop_layer
    orthogonal_nop_block
    orthogonal_attention_layer
