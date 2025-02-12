Custom Precision Quantization by Layer Name
===========================================

.. autofunction:: netspresso.quantizer.__init__.Quantizer.custom_precision_quantization_by_layer_name


Example
-------

.. code-block:: python

    from netspresso import NetsPresso
    from netspresso.enums import QuantizationPrecision


    netspresso = NetsPresso(email="YOUR_EMAIL", password="YOUR_PASSWORD")

    quantizer = netspresso.quantizer()

    recommendation_metadata = quantizer.get_recommendation_precision(
        input_model_path="./examples/sample_models/test.onnx",
        output_dir="./outputs/quantized/automatic_quantization",
        dataset_path="./examples/sample_datasets/pickle_calibration_dataset_128x128.npy",
        weight_precision=QuantizationPrecision.INT8,
        activation_precision=QuantizationPrecision.INT8,
        threshold=0,
    )
    recommendation_precisions = quantizer.load_recommendation_precision_result(recommendation_metadata.recommendation_result_path)

    quantization_result = quantizer.custom_precision_quantization_by_layer_name(
        input_model_path="./examples/sample_models/test.onnx",
        output_dir="./outputs/quantized/custom_precision_quantization_by_layer_name",
        dataset_path="./examples/sample_datasets/pickle_calibration_dataset_128x128.npy",
        precision_by_layer_name=recommendation_precisions.layers,
    )
