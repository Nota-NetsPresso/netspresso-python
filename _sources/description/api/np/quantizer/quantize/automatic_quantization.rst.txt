Automatic Quantization
======================

.. autofunction:: netspresso.quantizer.__init__.Quantizer.automatic_quantization


Example
-------

.. code-block:: python

    from netspresso import NetsPresso
    from netspresso.enums import QuantizationPrecision


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    quantizer = netspresso.quantizer()
    quantization_result = quantizer.automatic_quantization(
        input_model_path="./examples/sample_models/test.onnx",
        output_dir="./outputs/quantized/automatic_quantization",
        dataset_path="./examples/sample_datasets/pickle_calibration_dataset_128x128.npy",
        weight_precision=QuantizationPrecision.INT8,
        activation_precision=QuantizationPrecision.INT8,
        threshold=0,
    )
