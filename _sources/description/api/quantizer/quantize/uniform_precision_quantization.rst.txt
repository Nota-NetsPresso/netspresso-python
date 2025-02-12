Plain Quantization
======================

.. autofunction:: netspresso.quantizer.__init__.Quantizer.uniform_precision_quantization


Example
-------

.. code-block:: python

    from netspresso import NetsPresso
    from netspresso.enums import QuantizationPrecision, SimilarityMetric


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    quantizer = netspresso.quantizer()
    quantization_result = quantizer.uniform_precision_quantization(
        input_model_path="./examples/sample_models/test.onnx",
        output_dir="./outputs/quantized/uniform_precision_quantization",
        dataset_path="./examples/sample_datasets/pickle_calibration_dataset_128x128.npy",
        metric=SimilarityMetric.SNR,
        weight_precision=QuantizationPrecision.INT8,
        activation_precision=QuantizationPrecision.INT8,
    )
