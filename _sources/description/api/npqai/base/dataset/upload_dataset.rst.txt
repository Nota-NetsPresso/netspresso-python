Upload Dataset
==============

.. autofunction:: netspresso.np_qai.base.NPQAIBase.upload_dataset


Example
-------

.. code-block:: python

    from netspresso import NPQAI

    QAI_HUB_API_TOKEN = "YOUR_QAI_HUB_API_TOKEN"
    np_qai = NPQAI(api_token=QAI_HUB_API_TOKEN)
    dataset = np_qai.upload_dataset("YOUR_DATASET_PATH")
