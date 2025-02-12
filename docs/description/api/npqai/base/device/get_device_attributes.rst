Get Device Attributes
=====================

.. autofunction:: netspresso.np_qai.base.NPQAIBase.get_device_attributes


Example
-------

.. code-block:: python

    from netspresso import NPQAI

    QAI_HUB_API_TOKEN = "YOUR_QAI_HUB_API_TOKEN"
    np_qai = NPQAI(api_token=QAI_HUB_API_TOKEN)
    device_attributes = np_qai.get_device_attributes()
