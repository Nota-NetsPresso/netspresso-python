Upload Model
============

.. autofunction:: netspresso.np_qai.base.NPQAIBase.upload_model


Example
-------

.. code-block:: python

    from netspresso import NPQAI

    QAI_HUB_API_TOKEN = "YOUR_QAI_HUB_API_TOKEN"
    np_qai = NPQAI(api_token=QAI_HUB_API_TOKEN)
    model = np_qai.upload_model("YOUR_MODEL_PATH")
